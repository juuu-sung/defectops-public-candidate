"""Read-only replay of selected A cases and exact normal-patch provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
from scripts import mpi_patch_trace_protocol as p
from scripts import mpi_patch_trace_cards as cards
from scripts import probe_mpi_scores as probe

fit, ev = p.fit, p.ev
CASE_FILES = ["sample.json", "trace.json", "card-1.jpg", "card-2.jpg"]


def feature_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


@contextmanager
def capture_neighbors(core, captured):
    """Observe the library result, without replacing its distance/search algorithm."""
    original = core.nearest_neighbors
    had_override = "nearest_neighbors" in vars(core)
    override = vars(core).get("nearest_neighbors")

    def record(embedding, n_neighbors):
        result = original(embedding, n_neighbors)
        if n_neighbors == 1 and tuple(embedding.shape) == (4096, 768):
            captured.append((result[0].detach().cpu().numpy(), result[1].detach().cpu().numpy(),
                             feature_digest(embedding.detach().cpu().numpy())))
        return result

    core.nearest_neighbors = record
    try:
        yield
    finally:
        if had_override:
            core.nearest_neighbors = override
        else:
            del core.nearest_neighbors


class TraceRuntime(fit.TorchRuntime):
    def bank_fingerprint(self):
        bank = self.model.model.memory_bank
        if list(bank.shape) != p.POLICY["bank_shape"] or bank.dtype != self.torch.float32:
            raise ValueError("Frozen bank shape/dtype differs")
        h = hashlib.sha256()
        for start in range(0, len(bank), 1024):
            array = bank[start:start+1024].detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite memory bank")
            h.update(np.ascontiguousarray(array, dtype="<f4").tobytes())
        return {"shape": list(bank.shape), "float32_le_sha256": h.hexdigest()}

    def trace(self, image):
        captured = []
        with capture_neighbors(self.model.model, captured):
            raw, values = self.predict(image)
        if len(captured) != 1:
            raise ValueError("Expected exactly one native 4096-patch nearest-neighbor call")
        distances, indices, digest = captured[0]
        if indices.shape != (4096,) or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= 78643):
            raise ValueError("Invalid selected bank indices")
        return raw, values, distances, indices, digest

    def bank_hashes(self, indices):
        ids = self.torch.tensor(indices, dtype=self.torch.long, device=self.model.model.memory_bank.device)
        return p.vector_hashes(self.model.model.memory_bank.index_select(0, ids).detach().cpu().numpy())

    def embeddings(self, image):
        core = self.model.model
        if core.training or core.feature_extractor.training or core.tiler is not None:
            raise ValueError("Feature replay requires frozen eval extractor with no tiler")
        with self.torch.no_grad():
            features = core.feature_extractor(fit._preprocess(image, 256, self.gpu))
            pooled = {layer: core.feature_pooler(value) for layer, value in features.items()}
            embedding = core.generate_embedding(pooled)
            if tuple(embedding.shape) != (1, 768, 64, 64):
                raise ValueError("Unexpected feature replay grid")
            return core.reshape_embedding(embedding)

    def verify_feature_path(self, image):
        _, _, _, _, native_sha = self.trace(image)
        direct = self.embeddings(image).detach().cpu().numpy()
        if feature_digest(direct) != native_sha:
            raise ValueError("Direct feature replay differs from native inference embeddings")
        return {"native_embedding_sha256": native_sha, "direct_replay_exact": True}

    def candidate_features(self, image):
        embedding = self.embeddings(image)
        indices = fit.spatial_patch_indices(64, 2)
        chosen = self.torch.tensor(indices, dtype=self.torch.long, device=embedding.device)
        # Only one tile's 1024 candidates reside on CPU; no global candidate tensor or coreset.
        return embedding.index_select(0, chosen).detach().cpu().numpy(), indices


def create_case(source, photo, expected, reason, contract, runtime, recovered, stage):
    images, arrays, tiles, rows = [], [], [], []
    normal = photo["key"].startswith("calibration/")
    masks = [np.zeros((256, 256), dtype=bool) for _ in photo["regions"]] if normal else ev.read_masks(source, photo)
    # A damaged-mask photo must have no GT-based query selection anywhere in it.
    if photo["key"] == "defect/0006":
        masks = [None] * 12
    started = time.perf_counter()
    try:
        for row, mask in zip(photo["regions"], masks, strict=True):
            image = fit.checked_image(source, row) if normal else ev.read_tile(source, row, recovered)
            images.append(image)
            raw, values, distances, indices, embedding_sha = runtime.trace(image)
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid native map")
            arrays.append(values)
            top1 = fit.candidate_image_score(raw, values, fit.CANDIDATE)
            rows.append({"region": row["region"], "member": row["member"], "top1": top1, "native_raw": float(raw)})
            queries = p.select_query_cells(distances, mask)
            bank_ids = [int(indices[q["flat_index"]]) for q in queries]
            for q, index, h in zip(queries, bank_ids, runtime.bank_hashes(bank_ids), strict=True):
                q.update(bank_index=index, bank_vector_sha256=h)
            tiles.append({"region": row["region"], "member": row["member"], "queries": queries,
                          "native_embedding_sha256": embedding_sha,
                          "GT_available": mask is not None, "GT_pixels": int(mask.sum()) if mask is not None else None})
        sample = {k: photo[k] for k in ("key", "photo_id", "label")} | {"regions": rows}
        replay = p.check_replay(sample, expected, contract)
        for page in range(2):
            # The selected real cohort has no damaged GT. Keep this fail-closed if changed.
            if any(mask is None for mask in masks):
                raise ValueError("Incomplete GT cannot be rendered as a normal/empty mask")
            probe.cards.render(sample, images, arrays, masks, page, ";".join(reason)).save(stage / f"card-{page+1}.jpg", quality=90)
        trace = {"key": photo["key"], "label": photo["label"], "reasons": reason, "replay": replay,
                 "tiles": tiles, "seconds_including_cards": time.perf_counter()-started}
        fit.write_json(stage / "sample.json", sample)
        fit.write_json(stage / "trace.json", trace)
    finally:
        for image in images:
            image.close()


def read_case(root, name, signature, photo, expected, reason, contract):
    receipt = fit.read_phase(root, name, signature, CASE_FILES)
    if receipt["input_photo_sha256"] != fit.digest(photo):
        raise ValueError("Cached photo input differs")
    sample = json.loads((root / name / "sample.json").read_bytes())
    trace = json.loads((root / name / "trace.json").read_bytes())
    if (trace["key"] != photo["key"] or trace["reasons"] != reason
            or trace["replay"] != p.check_replay(sample, expected, contract)
            or [r["region"] for r in trace["tiles"]] != fit.REGIONS):
        raise ValueError("Cached trace identity/replay differs")
    return trace, sample, receipt


def result_files(selection):
    names = ["diagnosis.json", "plan.json", "samples.json", "queries.json", "provenance.json", "A-calibration.json", "index.html", "handoff-manifest.json"]
    for key in selection:
        prefix = key.replace("/", "-")
        names.extend(f"cards/{prefix}-{suffix}.jpg" for suffix in ("1", "2", "neighbors"))
    return names


def execute(root, source, checkpoint, recovered, config, preparation, records, photos, expected,
            selection, contract, identity, git, handoff, runtime_factory=TraceRuntime):
    signature = fit.digest(identity)
    case_ids = [f"case-{i+1:02d}" for i in range(len(photos))]
    train_ids = preparation["training_photo_ids"]
    train_names = [f"normal-{i+1:02d}" for i in range(len(train_ids))]
    with fit.locked_run(root):
        if (root / "integrity-block.json").exists():
            raise ValueError("Prior bank integrity failure; preserve this run and request review before proceeding")
        if not (root / "identity").exists():
            if any((root / n).exists() for n in ("cases", "sources", "result")):
                raise ValueError("Outputs without input identity; preserved")
            with fit.phase_stage(root, "identity") as stage:
                fit.write_json(stage / "plan.json", {"identity": identity, "selection": selection, "git": git})
                fit.seal_phase(stage, "identity", signature, {}, ["plan.json"])
        fit.read_phase(root, "identity", signature, ["plan.json"])
        case_root, source_root = root / "cases", root / "sources"
        case_root.mkdir(exist_ok=True)
        source_root.mkdir(exist_ok=True)
        cached = {}
        for name, photo in zip(case_ids, photos, strict=True):
            if (case_root / name).exists():
                cached[name] = read_case(case_root, name, signature, photo, expected[photo["key"]], selection[photo["key"]], contract)
        cached_sources = {}
        for i, name in enumerate(train_names):
            if (source_root / name).exists():
                receipt = fit.read_phase(source_root, name, signature, ["origins.json"])
                chunk = json.loads((source_root / name / "origins.json").read_bytes())
                if chunk["photo_id"] != train_ids[i] or receipt["training_rows_sha256"] != fit.digest(preparation["training_rows"][i*12:(i+1)*12]):
                    raise ValueError("Cached training source identity differs")
                cached_sources[name] = chunk, receipt
        if cached_sources and len(cached) != len(photos):
            raise ValueError("Source trace exists without all completed query cases")
        if cached_sources:
            cached_target_sha = fit.digest(p.targets_from_cases([cached[n][0] for n in case_ids]))
            if any(receipt["targets_sha256"] != cached_target_sha for _, receipt in cached_sources.values()):
                raise ValueError("Cached source search targets differ")
        if (root / "result").exists():
            fit.read_phase(root, "result", signature, result_files(selection))
            if len(cached) != len(photos) or len(cached_sources) != len(train_ids):
                raise ValueError("Completed diagnosis is missing verified phase cache")
            probe.export(root / "result", handoff)
            return json.loads((root / "result/diagnosis.json").read_bytes())
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        runtime = None
        try:
            gate_sha = None
            if len(cached) < len(photos) or len(cached_sources) < len(train_ids):
                print("Loading saved model for bounded diagnosis. No fitting or threshold change.", flush=True)
                runtime = runtime_factory(config)
                runtime.load(checkpoint, records["fit.json"]["checkpoint_context"])
                if runtime.environment() != records["reload.json"]["receipt"]["runtime"]:
                    raise ValueError("Runtime environment differs from reviewed checkpoint replay")
                before_bank = runtime.bank_fingerprint()
                gate = ev.normal_probe(source, preparation, records["reload.json"]["photos"], runtime)
                if not gate["passed"]:
                    fit.write_json(attempt / "blocked-probe.json", gate)
                    raise ValueError(f"Normal replay gate failed: {gate['reasons']}")
                with fit.checked_image(source, preparation["training_rows"][0]) as image:
                    feature_check = runtime.verify_feature_path(image)
                gate.update(bank=before_bank, feature_path=feature_check, process_id=os.getpid(), runtime=runtime.environment())
                fit.write_json(attempt / "normal-probe.json", gate)
                gate_sha = fit.sha256_file(attempt / "normal-probe.json")
            for name, photo in zip(case_ids, photos, strict=True):
                if name not in cached:
                    with fit.phase_stage(case_root, name) as stage:
                        create_case(source, photo, expected[photo["key"]], selection[photo["key"]], contract, runtime, recovered, stage)
                        fit.seal_phase(stage, name, signature, {"input_photo_sha256": fit.digest(photo), "normal_probe_sha256": gate_sha}, CASE_FILES)
                    cached[name] = read_case(case_root, name, signature, photo, expected[photo["key"]], selection[photo["key"]], contract)
                print(f"Diagnostic case {name}: {photo['key']}", flush=True)
            cases = [cached[n][0] for n in case_ids]
            targets = p.targets_from_cases(cases)
            target_sha = fit.digest(targets)
            for _, receipt in cached_sources.values():
                if receipt["targets_sha256"] != target_sha:
                    raise ValueError("Cached source search targets differ")
            for i, name in enumerate(train_names):
                if name not in cached_sources:
                    rows = preparation["training_rows"][i*12:(i+1)*12]
                    with fit.phase_stage(source_root, name) as stage:
                        started = time.perf_counter()
                        origins, count = [], 0
                        for row in rows:
                            with fit.checked_image(source, row) as image:
                                features, indices = runtime.candidate_features(image)
                            if list(features.shape) != [1024, config["feature_dimension"]] or indices != fit.spatial_patch_indices(64, 2):
                                raise ValueError("Original stride-2 training candidate layout differs")
                            origins.extend(p.find_origins(features, indices, row, set(targets.values())))
                            count += len(indices)
                            del features
                        chunk = {"photo_id": train_ids[i], "candidate_embeddings_checked": count, "origins": origins,
                                 "feature_replay_seconds": time.perf_counter()-started}
                        fit.write_json(stage / "origins.json", chunk)
                        receipt = fit.seal_phase(stage, name, signature, {"training_rows_sha256": fit.digest(rows),
                            "targets_sha256": target_sha, "normal_probe_sha256": gate_sha}, ["origins.json"])
                    cached_sources[name] = chunk, receipt
                if sum(len(c[0]["origins"]) for c in cached_sources.values()) > p.POLICY["maximum_origin_records"]:
                    raise ValueError("Excessive exact-source ambiguity; records preserved, no arbitrary truncation")
                print(f"Original normal feature replay {i+1}/{len(train_ids)} (no coreset selection)", flush=True)
            if runtime is not None:
                if runtime.bank_fingerprint() != before_bank:
                    fit.write_json(root / "integrity-block.json", {"reason": "bank_mutated", "attempt": attempt.name})
                    raise ValueError("In-memory bank changed during diagnosis")
                fit.write_json(attempt / "runtime.json", {"process_id": os.getpid(), "bank_unchanged": True, "memory": runtime.memory()})
            receipts = [v[2] for v in cached.values()] + [v[1] for v in cached_sources.values()]
            hashes = {r["normal_probe_sha256"] for r in receipts}
            gates = [json.loads(f.read_bytes()) for f in sorted(root.glob("attempt-*/normal-probe.json")) if fit.sha256_file(f) in hashes]
            if len(gates) != len(hashes) or not all(g["passed"] and g["feature_path"]["direct_replay_exact"] for g in gates):
                raise ValueError("Cached phases missing passing normal/feature gate")
            if len({fit.digest(g["bank"]) for g in gates}) != 1:
                raise ValueError("Bank changed across resumed attempts")
            chunks = [cached_sources[n][0] for n in train_names]
            provenance = p.summarize_origins(targets, chunks, cases)
            unresolved = provenance["unresolved_bank_indices"]
            report = {"status": "patch_trace_completed_with_unresolved_sources" if unresolved else "patch_trace_completed_exploratory",
                "execution_sha256": signature, "policy": p.POLICY, "case_count": len(cases),
                "training_original_photos_replayed": len(train_ids),
                "training_candidates_checked": sum(c["candidate_embeddings_checked"] for c in chunks),
                "selected_bank_vectors": len(targets), "resolved_bank_vectors": provenance["resolved_bank_vectors"],
                "ambiguous_bank_vectors": provenance["ambiguous_bank_vectors"], "unresolved_bank_vectors": len(unresolved),
                "maximum_absolute_score_replay_difference": max(c["replay"]["maximum_absolute_score_difference"] for c in cases),
                "normal_probes": gates, "mixed_attempts": len(gates)>1,
                "runtime_attempts": [json.loads(f.read_bytes()) for f in sorted(root.glob("attempt-*/runtime.json"))],
                "interrupted_attempts": [json.loads(f.read_bytes()) for f in sorted(root.glob("attempt-*/failure.json"))],
                "training_performed": False, "threshold_changed": False, "automatic_pass_allowed": False,
                "service_model_changed": False, "confirmatory_claim_allowed": False,
                "conclusion": "Evidence for human review, not an automatic root-cause verdict or new performance result."}
            with fit.phase_stage(root, "result") as stage:
                (stage / "cards").mkdir()
                payload = {"diagnosis.json": report, "queries.json": cases, "provenance.json": provenance,
                           "samples.json": [cached[n][1] for n in case_ids], "A-calibration.json": contract,
                           "plan.json": json.loads((root / "identity/plan.json").read_bytes())}
                for name, value in payload.items():
                    fit.write_json(stage / name, value)
                for name, photo, case in zip(case_ids, photos, cases, strict=True):
                    prefix = photo["key"].replace("/", "-")
                    for page in (1, 2):
                        os.link(case_root / name / f"card-{page}.jpg", stage / "cards" / f"{prefix}-{page}.jpg")
                    cards.render_neighbors(source, photo, case, provenance, preparation, recovered, contract, cached[name][1]).save(
                        stage / "cards" / f"{prefix}-neighbors.jpg", quality=90)
                (stage / "index.html").write_text(cards.index(report, cases, provenance), encoding="utf-8")
                names = result_files(selection)
                fit.write_json(stage / "handoff-manifest.json", {n: fit.sha256_file(stage/n) for n in names if n != "handoff-manifest.json"})
                fit.seal_phase(stage, "result", signature, {}, names)
            probe.export(root / "result", handoff)
            return report
        except BaseException as exc:
            fit.write_json(attempt / "failure.json", {"error": f"{type(exc).__name__}: {exc}",
                "completed_cases": len(cached), "completed_source_photos": len(cached_sources), "process_id": os.getpid(),
                "automatic_pass_allowed": False, "note": "Old data/checkpoint/calibration unchanged. Completed phases preserved."})
            print(f"Failure record: {attempt / 'failure.json'}", file=sys.stderr)
            raise
        finally:
            if runtime is not None:
                runtime.release()


def run(project, archive):
    handoffs = project / "artifacts/handoff"
    config = fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    preparation = fit.load_preparation(handoffs / "mpi-pcb-normal-bank-preparation-v1.zip", config)
    records = ev.load_fit(handoffs / "mpi-pcb-normal-bank-fit-v1.zip", project, config, preparation)
    previous, _ = ev.load_prior(project, handoffs)
    baseline = ev.checked_bundle(handoffs / "mpi-pcb-normal-bank-evaluation-v1.zip", p.score.EVALUATION_SHA256)
    score_bundle = ev.checked_bundle(handoffs / "mpi-pcb-score-probe-v1.zip", p.SCORE_ZIP_SHA256)
    if score_bundle["baseline-scores.json"] != baseline["scores.json"]:
        raise ValueError("Reviewed score experiment baseline differs")
    for name, h in score_bundle["plan.json"]["identity"]["implementation"]["source_files_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != h:
            raise ValueError(f"Previous score-probe implementation changed: {name}")
    contract = score_bundle["calibration.json"]
    if p.score.calibration_contract(score_bundle["raw-scores.json"]["development"], score_bundle["raw-scores.json"]["calibration"]) != contract:
        raise ValueError("Previously frozen A calibration does not reproduce")
    selection = p.select_cases(score_bundle, baseline["cases.json"])
    if len(selection) != 12:
        raise ValueError("Reviewed 12-case diagnostic selection changed")
    normals = probe.normal_photos(preparation["calibration_photo_ids"], preparation["calibration_rows"], "calibration")
    by_key = {s["key"]: s for s in normals + previous["test-audit.json"]["photos"]}
    photos = [by_key[k] for k in selection]
    expected = {s["key"]: s for group in ("calibration", "test") for s in score_bundle["raw-scores.json"][group]}
    checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256 or fit.sha256_file(archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Source archive/checkpoint changed; no repair or replacement attempted")
    recovered = ev.recovered_bytes(project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png", previous)
    names = ["trace_mpi_patches.py", "mpi_patch_trace_protocol.py", "mpi_patch_trace_cards.py", "probe_mpi_scores.py", "mpi_score_probe_protocol.py", "mpi_score_probe_cards.py"]
    implementation = ev.run_identity(project)
    implementation["source_files_lf_sha256"].update({f"scripts/{n}": hashlib.sha256((project/"scripts"/n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in names})
    identity = {"policy": p.POLICY, "implementation": implementation, "selection": selection,
                "photos_sha256": fit.digest(photos), "A_contract_sha256": fit.digest(contract)}
    with zipfile.ZipFile(archive) as source:
        return execute(project / "artifacts/analyses" / p.NAME, source, checkpoint, recovered, config, preparation,
            records, photos, expected, selection, contract, identity, fit.git_state(project), handoffs / f"{p.NAME}.zip")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.project.resolve(), args.archive.resolve())
        print(json.dumps({k: report[k] for k in ("status", "case_count", "training_candidates_checked", "selected_bank_vectors",
            "resolved_bank_vectors", "ambiguous_bank_vectors", "unresolved_bank_vectors", "maximum_absolute_score_replay_difference")}, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
