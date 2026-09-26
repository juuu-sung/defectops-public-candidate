"""Fit the reviewed MPI normal bank and calibrate normals; never evaluate test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.padim_experiment import git_state
from defectops.realiad_followup_development import _seed_everything, candidate_image_score
from defectops.realiad_followup_final import _gpu_modules, _model, _predict, _preprocess
from defectops.visa_pcb_normal_bank_development import spatial_patch_indices
from scripts.prepare_mpi_pcb_normal_bank import (
    EXPECTED_ARCHIVE_SHA256, EXPECTED_V4_SHA256, REGIONS, canonical, digest,
    load_config, patch_budget, sha256_file,
)
from scripts.preflight_mpi_pcb_models import EXPECTED_IDS_SHA256

PREPARATION_SHA256 = "689716c33cc8df1308b985d5d1b6c76a13f451f6499cccfd349553b3dc23965b"
TRAIN_IDS_SHA256 = "2bbfbbd56016c712004d3ce3f1bd5e5a67740c9917ed46fd34b36fb866eea2c8"
CANDIDATE = {"image_score": "mean_of_top_map_fraction", "top_fraction": 0.01}
RELOAD_TOLERANCE = 1e-4
FINAL_STATUS = "mpi_pcb_target_bank_fit_calibrated_exploratory"


def load_preparation(path: Path, config: dict) -> dict:
    if sha256_file(path) != PREPARATION_SHA256:
        raise ValueError("Preparation ZIP is not the reviewed 64-photo cohort")
    with zipfile.ZipFile(path) as source:
        names = {"preparation.json", "normal-fingerprints.json", "config.json", "handoff-manifest.json"}
        if len(source.namelist()) != len(names) or set(source.namelist()) != names:
            raise ValueError("Unexpected preparation ZIP members")
        raw = {name: source.read(name) for name in names}
    manifest = json.loads(raw["handoff-manifest.json"])
    if set(manifest) != names - {"handoff-manifest.json"} or any(
        hashlib.sha256(raw[name]).hexdigest() != value for name, value in manifest.items()
    ):
        raise ValueError("Preparation internal checksum differs")
    report = json.loads(raw["preparation.json"])
    identity = report["input_identity"]
    if not (
        json.loads(raw["config.json"]) == config
        and report["status"] == "normal_bank_preparation_ready_for_review"
        and identity["config_sha256"] == digest(config)
        and identity["archive_sha256"] == EXPECTED_ARCHIVE_SHA256
        and identity["calibration_v4_sha256"] == EXPECTED_V4_SHA256
        and digest(report["training_photo_ids"]) == TRAIN_IDS_SHA256
        and digest(report["calibration_photo_ids"]) == EXPECTED_IDS_SHA256
        and report["planned_patch_budget"] == patch_budget(config)
        and report["partial_tile_duplicate_groups_in_fit"] == []
        and report["training_performed"] is False
        and report["automatic_pass_allowed"] is False
    ):
        raise ValueError("Preparation contract differs")
    for name, count in (("training", 64), ("calibration", 169)):
        ids = report[f"{name}_photo_ids"]
        rows = report[f"{name}_rows"]
        if len(ids) != count or len(set(ids)) != count or len(rows) != count * 12:
            raise ValueError("Preparation photo/region count differs")
        if [(row["photo_id"], row["region"]) for row in rows] != [(p, r) for p in ids for r in REGIONS]:
            raise ValueError("Preparation row order differs")
        for row in rows:
            if row["member"] != f'extracted/{row["region"]}/train/good/{row["photo_id"]}.png':
                raise ValueError("Non-training path in preparation")
    train_rgb = {row["decoded_rgb_sha256"] for row in report["training_rows"]}
    cal_rgb = {row["decoded_rgb_sha256"] for row in report["calibration_rows"]}
    if len(train_rgb) != 768 or train_rgb & cal_rgb:
        raise ValueError("Preparation exact train/calibration duplicate")
    return report


def checked_image(source: zipfile.ZipFile, row: dict) -> Image.Image:
    name = f'extracted/{row["region"]}/train/good/{row["photo_id"]}.png'
    if (row["region"] not in REGIONS or row["member"] != name
        or len(row["photo_id"]) != 4 or not row["photo_id"].isascii() or not row["photo_id"].isdigit()):
        raise ValueError("Refusing non-training image access")
    info = source.getinfo(name)
    if info.file_size > 8 * 1024 * 1024:
        raise ValueError("Unexpected PNG size")
    raw = source.read(info)
    if hashlib.sha256(raw).hexdigest() != row["member_sha256"]:
        raise ValueError(f"Normal bytes changed: {name}")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
            raise ValueError(f"Expected RGB PNG 256x256: {name}")
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if hashlib.sha256(image.tobytes()).hexdigest() != row["decoded_rgb_sha256"]:
            raise ValueError(f"Normal decoded pixels changed: {name}")
        return image.copy()


def retain_spatial_patches(store: list, index: int, indices, config: dict) -> None:
    expected_shape = (config["feature_grid_side"] ** 2, config["feature_dimension"])
    if len(store) != index or tuple(store[-1].shape) != expected_shape:
        raise ValueError("Anomalib embedding layout differs from the frozen spatial plan")
    store[-1] = store[-1].index_select(0, indices)


class TorchRuntime:
    """The only GPU boundary. Tests use an explicitly fake, non-service runtime."""

    def __init__(self, config: dict):
        self.config = config
        self.gpu = _gpu_modules()
        self.torch = self.gpu[0]
        self.model = None
        self.recipe = {"base_model": config}
        _seed_everything(config["seed"], self.torch)

    def fit(self, images):
        torch, _, Patchcore, _, device = self.gpu
        cfg = self.config
        self.model = Patchcore(
            backbone=cfg["backbone"], layers=cfg["layers"], pre_trained=True,
            coreset_sampling_ratio=cfg["coreset_sampling_ratio"], num_neighbors=cfg["num_neighbors"],
            pre_processor=False, post_processor=False, evaluator=False, visualizer=False,
        ).to(device)
        self.model.model.train()
        self.model.model.feature_extractor.eval()
        indices = torch.tensor(spatial_patch_indices(cfg["feature_grid_side"], cfg["spatial_patch_stride"]),
                               dtype=torch.long, device=device)
        self.reset_peak()
        started = time.perf_counter()
        count = 0
        with torch.no_grad():
            for count, image in enumerate(images, start=1):
                try:
                    self.model.model(_preprocess(image, cfg["image_size"], self.gpu))
                    if self.model.model.embedding_store[-1].dtype != torch.float32:
                        raise ValueError("Expected float32 embeddings for the frozen memory plan")
                    retain_spatial_patches(self.model.model.embedding_store, count, indices, cfg)
                finally:
                    image.close()
                if count % 25 == 0 or count == patch_budget(cfg)["training_tiles"]:
                    print(f"MPI normal fit {count}/{patch_budget(cfg)['training_tiles']}", flush=True)
        self.synchronize()
        extraction_seconds = time.perf_counter() - started
        budget = patch_budget(cfg)
        candidates = sum(int(value.shape[0]) for value in self.model.model.embedding_store)
        if count != budget["training_tiles"] or candidates != budget["candidate_embeddings"]:
            raise ValueError("Candidate pool differs; no automatic shrinking allowed")
        print(f"Selecting 10% coreset: {candidates} candidates -> {budget['coreset_embeddings']} representatives. This may take hours.", flush=True)
        started = time.perf_counter()
        self.model.fit()
        self.model._is_fitted.fill_(True)
        self.model.model.eval()
        self.synchronize()
        coreset_seconds = time.perf_counter() - started
        if list(self.model.model.memory_bank.shape) != [budget["coreset_embeddings"], cfg["feature_dimension"]]:
            raise ValueError("Coreset shape differs from the frozen budget")
        return {"feature_extraction_seconds": extraction_seconds, "coreset_seconds": coreset_seconds,
                "candidate_embeddings": candidates, "memory_bank_shape": list(self.model.model.memory_bank.shape),
                "memory": self.memory(), "runtime": self.environment()}

    def save(self, path: Path, context: dict):
        state = {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
        self.torch.save({"format_version": 1, "category": context["category"], "layers": self.config["layers"],
                         "model_seed": context["seed"], "manifest_sha256": context["manifest_sha256"],
                         "execution_config_sha256": context["execution_config_sha256"],
                         "memory_bank_shape": list(self.model.model.memory_bank.shape), "state_dict": state}, path)

    def load(self, path: Path, context: dict):
        self.release()
        self.model = _model(self.recipe, path, context, self.gpu)
        self.reset_peak()

    def predict(self, image):
        return _predict(self.model, image, self.recipe, self.gpu)

    def synchronize(self):
        self.torch.cuda.synchronize()

    def reset_peak(self):
        self.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def memory(self):
        self.synchronize()
        try:
            import psutil
            rss = psutil.Process().memory_info().rss
        except ImportError:
            rss = None
        return {"torch_cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "torch_cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
                "process_rss_sample_bytes": rss,
                "scope": "PyTorch allocator peaks for this phase, not total GPU usage; RSS is a sample, not peak RAM"}

    def environment(self):
        import importlib.metadata
        return {"kind": "real_cuda", "python": platform.python_version(), "platform": platform.platform(),
                "gpu": self.torch.cuda.get_device_name(), "cuda_runtime": self.torch.version.cuda,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "anomalib")}}

    def release(self):
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()


def score_normals(source, rows, photo_ids, runtime, label):
    expected = [(p, r) for p in photo_ids for r in REGIONS]
    if [(row["photo_id"], row["region"]) for row in rows] != expected:
        raise ValueError("Normal calibration row order differs")
    photos = []
    runtime.reset_peak()
    started = time.perf_counter()
    latencies = []
    for index, p in enumerate(photo_ids):
        regions = []
        for row in rows[index * len(REGIONS):(index + 1) * len(REGIONS)]:
            runtime.synchronize()
            before = time.perf_counter()
            with checked_image(source, row) as image:
                raw, anomaly_map = runtime.predict(image)
            values = np.asarray(anomaly_map)
            if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid normal anomaly map")
            score = candidate_image_score(raw, values, CANDIDATE)
            runtime.synchronize()
            elapsed_ms = (time.perf_counter() - before) * 1000
            latencies.append(elapsed_ms)
            regions.append({"region": row["region"], "member": row["member"], "score": score})
        photos.append({"photo_id": p, "score": max(row["score"] for row in regions), "regions": regions})
        if (index + 1) % 10 == 0 or index + 1 == len(photo_ids):
            print(f"{label} {index + 1}/{len(photo_ids)} original photos", flush=True)
    runtime.synchronize()
    return photos, {"seconds": time.perf_counter() - started, "memory": runtime.memory(),
                    "tile_latency_ms": {"count": len(latencies), "mean": float(np.mean(latencies)),
                                         "p95_higher": float(np.quantile(latencies, .95, method="higher")),
                                         "scope": "ZIP read, checksum, PNG decode, preprocessing, GPU inference, map transfer and score; no model load; not a dedicated warm benchmark"}}


def validate_photos(photos, ids):
    if len(photos) != len(ids) or [p["photo_id"] for p in photos] != ids:
        raise ValueError("Saved normal photo order differs")
    for photo in photos:
        rows = photo["regions"]
        if [r["region"] for r in rows] != REGIONS or any(
            r["member"] != f'extracted/{r["region"]}/train/good/{photo["photo_id"]}.png'
            or not isinstance(r["score"], (int, float)) or not math.isfinite(r["score"]) or r["score"] < 0
            for r in rows
        ) or photo["score"] != max(r["score"] for r in rows):
            raise ValueError("Saved normal scores/paths differ")


def threshold(photos):
    return float(np.quantile([p["score"] for p in photos], .95, method="higher"))


def reload_differences(first, second, ids):
    validate_photos(first, ids)
    validate_photos(second, ids)
    photo_max = max(abs(a["score"] - b["score"]) for a, b in zip(first, second, strict=True))
    region_max = max(abs(a["score"] - b["score"]) for p, q in zip(first, second, strict=True)
                     for a, b in zip(p["regions"], q["regions"], strict=True))
    delta = abs(threshold(first) - threshold(second))
    if max(photo_max, region_max, delta) > RELOAD_TOLERANCE:
        raise ValueError(f"Reload score difference exceeds {RELOAD_TOLERANCE}: {photo_max}, {region_max}, {delta}")
    return {"maximum_photo_difference": photo_max, "maximum_region_difference": region_max,
            "threshold_difference": delta, "tolerance": RELOAD_TOLERANCE, "passed": True}


def write_json(path: Path, payload: dict):
    with path.open("xb") as target:
        target.write(canonical(payload))
        target.flush()
        os.fsync(target.fileno())


@contextmanager
def locked_run(root: Path):
    """OS lock releases on process exit; no stale-PID lock-file deletion needed."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another MPI normal-bank run is using this output") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def phase_stage(root: Path, phase: str):
    """Preserve incomplete attempts; only publish a fully written phase directory."""
    final = root / phase
    if final.exists():
        raise FileExistsError(f"Phase already exists: {final}")
    stage = Path(tempfile.mkdtemp(prefix=f".{phase}-attempt-", dir=root))
    try:
        yield stage
        if final.exists():
            raise FileExistsError(f"Phase appeared concurrently: {final}")
        os.rename(stage, final)
    except BaseException:
        print(f"Incomplete attempt preserved: {stage}. Completed phases are reusable.", file=sys.stderr)
        raise


def seal_phase(stage, phase, execution_sha, metadata, names):
    receipt = {"phase": phase, "execution_sha256": execution_sha, **metadata,
               "files": {name: sha256_file(stage / name) for name in names}}
    receipt["receipt_sha256"] = digest(receipt)
    write_json(stage / "receipt.json", receipt)
    return receipt


def read_phase(root, phase, execution_sha, names):
    path = root / phase
    try:
        receipt = json.loads((path / "receipt.json").read_bytes())
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_sha256")
        if (claimed != digest(unsigned) or receipt["phase"] != phase
            or receipt["execution_sha256"] != execution_sha or set(receipt["files"]) != set(names)
            or any(sha256_file(path / n) != receipt["files"][n] for n in names)):
            raise ValueError("Phase identity or checksum mismatch")
        return receipt
    except (ValueError, KeyError, OSError) as exc:
        raise ValueError(f"Existing {phase} is incomplete or changed; it was preserved: {path}") from exc


def execution_identity(project, config):
    relative = ["scripts/fit_mpi_pcb_normal_bank.py", "scripts/prepare_mpi_pcb_normal_bank.py",
                "src/defectops/realiad_followup_final.py", "src/defectops/realiad_followup_development.py",
                "src/defectops/visa_pcb_normal_bank_development.py", "src/defectops/padim_experiment.py"]
    return {"schema_version": 1, "preparation_zip_sha256": PREPARATION_SHA256,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256, "config_sha256": digest(config),
            "training_ids_sha256": TRAIN_IDS_SHA256, "calibration_ids_sha256": EXPECTED_IDS_SHA256,
            "reload_tolerance": RELOAD_TOLERANCE,
            "source_files_lf_sha256": {p: hashlib.sha256((project / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in relative}}


def export_handoff(root, output, report):
    payloads = {"input-lock.json": (root / "input-lock.json").read_bytes(), "fit.json": canonical(report),
                "fit-metadata.json": (root / "fit/receipt.json").read_bytes()}
    for phase in ("calibration", "reload"):
        payloads[f"{phase}.json"] = canonical({"receipt": json.loads((root / phase / "receipt.json").read_bytes()),
                                                "photos": json.loads((root / phase / "photos.json").read_bytes())})
    payloads["handoff-manifest.json"] = canonical({k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()})
    if output.exists():
        with zipfile.ZipFile(output) as existing:
            if len(existing.namelist()) != len(payloads) or set(existing.namelist()) != set(payloads) or any(
                existing.read(k) != value for k, value in payloads.items()
            ):
                raise ValueError("Existing handoff differs; preserved")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mpi-fit-handoff-", suffix=".zip", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for key, value in payloads.items():
                archive.writestr(key, value)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run_pipeline(root, archive_path, preparation, config, execution, git, handoff, runtime_factory=TorchRuntime):
    signature = digest(execution)
    budget = patch_budget(config)
    context = {"category": "mpi_pcb", "seed": config["seed"], "manifest_sha256": PREPARATION_SHA256,
               "execution_config_sha256": signature}
    runtime = None
    with locked_run(root):
        lock = root / "input-lock.json"
        if lock.exists():
            recorded = json.loads(lock.read_bytes())
            if recorded["execution"] != execution or recorded["config"] != config:
                raise ValueError("Existing execution/config differs; no output was changed")
        else:
            if any((root / phase).exists() for phase in ("fit", "calibration", "reload", "fit.json")):
                raise ValueError("Output has phases without input lock; preserved")
            # A staging input lock cannot leave a truncated final file after interruption.
            fd, tmp = tempfile.mkstemp(prefix=".input-lock-", dir=root)
            os.close(fd)
            temp = Path(tmp)
            try:
                with temp.open("wb") as target:
                    target.write(canonical({"execution": execution, "config": config, "git": git}))
                    target.flush()
                    os.fsync(target.fileno())
                os.link(temp, lock)
            finally:
                temp.unlink(missing_ok=True)
        completed = [(root / phase).exists() for phase in ("fit", "calibration", "reload")]
        if (completed[1] and not completed[0]) or (completed[2] and not completed[1]) or (
            (root / "fit.json").exists() and not all(completed)
        ):
            raise ValueError("Completed phases are missing or out of order; preserved without GPU work")
        try:
            did_fit = not (root / "fit").exists()
            if did_fit:
                print("1/4 Fitting reviewed MPI normal bank; no test data...", flush=True)
                runtime = runtime_factory(config)
                with phase_stage(root, "fit") as stage, zipfile.ZipFile(archive_path) as source:
                    result = runtime.fit(checked_image(source, row) for row in preparation["training_rows"])
                    if (result["memory_bank_shape"] != [budget["coreset_embeddings"], config["feature_dimension"]]
                        or result["candidate_embeddings"] != budget["candidate_embeddings"]):
                        raise ValueError("Runtime fit does not match the frozen patch budget")
                    runtime.save(stage / "model.pt", context)
                    seal_phase(stage, "fit", signature, {**result, "context": context, "git": git}, ["model.pt"])
            fitted = read_phase(root, "fit", signature, ["model.pt"])
            if fitted["context"] != context or fitted["memory_bank_shape"] != [budget["coreset_embeddings"], config["feature_dimension"]] or fitted["candidate_embeddings"] != budget["candidate_embeddings"]:
                raise ValueError("Saved fit shape or context differs")
            checkpoint = root / "fit/model.pt"
            checkpoint_sha = fitted["files"]["model.pt"]
            ids = preparation["calibration_photo_ids"]
            for phase in ("calibration", "reload"):
                if not (root / phase).exists():
                    if runtime is None:
                        runtime = runtime_factory(config)
                    if phase == "reload" or not did_fit:
                        runtime.load(checkpoint, context)
                    print(f"{'2' if phase == 'calibration' else '3'}/4 {phase}: {len(ids)} normal photos x {len(REGIONS)} regions; no test data...", flush=True)
                    with phase_stage(root, phase) as stage, zipfile.ZipFile(archive_path) as source:
                        photos, timing = score_normals(source, preparation["calibration_rows"], ids, runtime, phase)
                        validate_photos(photos, ids)
                        metadata = {"checkpoint_sha256": checkpoint_sha, "image_threshold": threshold(photos),
                                    "normal_original_photos": len(ids), "normal_region_images": len(ids)*len(REGIONS),
                                    "timing": timing, "runtime": runtime.environment(), "git": git,
                                    "model_origin": "live_fit" if phase == "calibration" and did_fit else "loaded_checkpoint"}
                        if phase == "reload":
                            before = json.loads((root / "calibration/photos.json").read_bytes())
                            metadata["comparison"] = reload_differences(before, photos, ids)
                        write_json(stage / "photos.json", photos)
                        seal_phase(stage, phase, signature, metadata, ["photos.json"])
                receipt = read_phase(root, phase, signature, ["photos.json"])
                photos = json.loads((root / phase / "photos.json").read_bytes())
                validate_photos(photos, ids)
                if receipt["checkpoint_sha256"] != checkpoint_sha or receipt["image_threshold"] != threshold(photos) or receipt["normal_original_photos"] != len(ids) or receipt["normal_region_images"] != len(ids)*len(REGIONS):
                    raise ValueError(f"Saved {phase} contract differs")
            first = json.loads((root / "calibration/photos.json").read_bytes())
            second = json.loads((root / "reload/photos.json").read_bytes())
            comparison = reload_differences(first, second, ids)
            reload_receipt = read_phase(root, "reload", signature, ["photos.json"])
            if reload_receipt["comparison"] != comparison:
                raise ValueError("Saved reload comparison differs")
            report = {"schema_version": 1, "status": FINAL_STATUS, "category": "mpi_pcb",
                      "execution_sha256": signature, "preparation_zip_sha256": PREPARATION_SHA256,
                      "checkpoint_sha256": checkpoint_sha, "checkpoint_context": context,
                      "training_original_photos": len(preparation["training_photo_ids"]),
                      "training_region_images": len(preparation["training_rows"]),
                      "calibration_original_photos": len(ids), "calibration_region_images": len(ids)*len(REGIONS),
                      "memory_bank_shape": fitted["memory_bank_shape"], "candidate_embeddings": fitted["candidate_embeddings"],
                      "image_threshold": threshold(first), "threshold_rule": "95pct_higher_normal_photo_scores",
                      "region_score": config["region_score"], "photo_score": config["photo_score"],
                      "reload_verification": comparison,
                      "phase_receipts": {phase: json.loads((root / phase / "receipt.json").read_bytes())["receipt_sha256"] for phase in ("fit", "calibration", "reload")},
                      "test_images_decoded": 0, "test_masks_decoded": 0, "test_scores_computed": 0,
                      "test_previously_observed": True, "automatic_pass_allowed": False,
                      "confirmatory_claim_allowed": False, "pixel_threshold_calibrated": False,
                      "note": "Normal-only fit/calibration; not an evaluation or service deployment. Calibration-reload compares the committed normal scores with a freshly loaded checkpoint."}
            final = root / "fit.json"
            if final.exists():
                if json.loads(final.read_bytes()) != report:
                    raise ValueError("Existing final fit report differs; preserved")
            else:
                # Report is deterministic; publish atomically so a crash cannot truncate it.
                fd, name = tempfile.mkstemp(prefix=".fit-report-", dir=root)
                os.close(fd)
                temp = Path(name)
                try:
                    temp.write_bytes(canonical(report))
                    os.link(temp, final)
                finally:
                    temp.unlink(missing_ok=True)
            print("4/4 Exporting JSON-only fit handoff. Model remains on this machine.", flush=True)
            export_handoff(root, handoff, report)
            return report
        finally:
            if runtime is not None:
                runtime.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    try:
        config = load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
        prep = load_preparation(project / "artifacts/handoff/mpi-pcb-normal-bank-preparation-v1.zip", config)
        print("Checking original archive identity before GPU work...", flush=True)
        if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("MPI source archive checksum differs")
        execution = execution_identity(project, config)
        git = git_state(project)
        handoff = project / "artifacts/handoff/mpi-pcb-normal-bank-fit-v1.zip"
        report = run_pipeline(project / "artifacts/mpi-pcb-normal-bank-pilot-v1", args.archive, prep,
                              config, execution, git, handoff)
        print(json.dumps({key: report[key] for key in ("status", "training_original_photos", "calibration_original_photos",
                                                      "memory_bank_shape", "image_threshold", "reload_verification", "test_scores_computed")}
                         | {"handoff": str(handoff), "handoff_sha256": sha256_file(handoff)}, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Completed phases and incomplete attempts were preserved. Do not delete existing results.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
