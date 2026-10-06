"""Nested normal-only data scaling, not a new independent defect benchmark."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from scripts import mpi_supersimplenet_fit_protocol as original

fit = original.fit
NAME = "mpi-supersimplenet-scaling-v1"
ARMS = {"a64-e300": (64, 300), "c256-e75": (256, 75), "d256-e300": (256, 300)}
SOURCES = {
    "mpi-supersimplenet-fit-v1.zip": "ae4ccc9e7164e2a052efd1c3ef9be600f956f9fbab8eae7898e1f1cda73ce2e7",
    "mpi-normal-diversity-preparation-v1.zip": "ba5abc4f137d38022c9ded9e143ef9cc9b2523d2a53a80c95479c254a575ff82",
}
POOL_IDS_SHA = "5e318c663affb53405524ec40d7785b3f05811755434cef168f9ec6170a7c7eb"
POOL_ROWS_SHA = "31a890ca06b0096c9672173121de0e3f9ace96b33144ba384f2708f1982a70df"


def read_source(path, names):
    if fit.sha256_file(path) != SOURCES[path.name]:
        raise ValueError(f"Reviewed source ZIP missing or changed: {path.name}")
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate evidence ZIP member")
        return {name: json.loads(z.read(name)) for name in names}


def rank(pid):
    return hashlib.sha256(f"{NAME}\0additional-normal\0{pid}".encode()).hexdigest(), pid


def validate_group(group, count):
    ids, rows = group["photo_ids"], group["rows"]
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("Normal photo count or uniqueness differs")
    if [(r["photo_id"], r["region"]) for r in rows] != [(pid, r) for pid in ids for r in fit.REGIONS]:
        raise ValueError("Normal photo/region order differs")
    pixels = set()
    for row in rows:
        pid, region = row["photo_id"], row["region"]
        if (len(pid) != 4 or not pid.isascii() or not pid.isdigit()
                or row["member"] != f"extracted/{region}/train/good/{pid}.png"
                or row["width"] != 256 or row["height"] != 256):
            raise ValueError("Only original 256x256 normal tiles are permitted")
        for field in ("member_sha256", "decoded_rgb_sha256"):
            digest = row[field]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid normal fingerprint")
        if row["decoded_rgb_sha256"] in pixels:
            raise ValueError("Exact normal tile duplicate")
        pixels.add(row["decoded_rgb_sha256"])
    return set(ids), pixels


def select_nested(pool, cohorts):
    if (len(pool["ids"]) != 1086 or fit.digest(pool["ids"]) != POOL_IDS_SHA
            or fit.digest(pool["rows"]) != POOL_ROWS_SHA):
        raise ValueError("Reviewed eligible pool differs; no replacement or reroll")
    validate_group({"photo_ids": pool["ids"], "rows": pool["rows"]}, 1086)
    original.validate_cohorts(cohorts)
    by_id = {pid: pool["rows"][i*12:(i+1)*12] for i, pid in enumerate(pool["ids"])}
    base = cohorts["training"]
    if not set(base["photo_ids"]) <= set(by_id):
        raise ValueError("Original 64 must remain in eligible pool")
    if [r for pid in base["photo_ids"] for r in by_id[pid]] != base["rows"]:
        raise ValueError("Original 64 fingerprints changed")
    added = sorted(set(by_id)-set(base["photo_ids"]), key=rank)[:192]
    ids = base["photo_ids"] + added
    selected = {"photo_ids": ids, "rows": [r for pid in ids for r in by_id[pid]]}
    selected_ids, selected_pixels = validate_group(selected, 256)
    for role, count in (("development", 128), ("calibration", 169)):
        protected_ids, protected_pixels = validate_group(cohorts[role], count)
        if selected_ids & protected_ids or selected_pixels & protected_pixels:
            raise ValueError("Expanded training overlaps protected normal cohort")
    return selected


class Protocol:
    validate_predictions = staticmethod(original.validate_predictions)
    tile_scores = staticmethod(original.tile_scores)
    calibrate = staticmethod(original.calibrate)
    photo_scores = staticmethod(original.photo_scores)

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError("Unregistered scaling arm")
        self.arm = arm
        photos, epochs = ARMS[arm]
        tiles, per_epoch = photos*12, photos*12//8
        updates = per_epoch*epochs
        self.POLICY = deepcopy(original.POLICY)
        self.POLICY.update({
            "purpose": "normal_only_nested_data_scaling_research_not_service",
            "arm": arm, "normal_photos": photos, "tiles": tiles, "epochs": epochs,
            "updates_per_epoch": per_epoch, "updates": updates, "optimizer_updates": updates,
            "checkpoint_interval": per_epoch*10,
            "maximum_attempted_updates": updates+per_epoch*30,
            "selected_checkpoint": f"final_epoch{epochs}_step{updates}_only",
            "lr_drop_updates": [updates*8//10, updates*9//10],
            "lr_schedule": "multiply_by0.4_after_80_and_90_percent_of_registered_updates",
            "shuffle": "original formal-fit SHA256 salt; epoch and canonical training row index",
            "loss": "Anomalib 2.2.0 SSNLoss unchanged and public training forward",
            "free_disk_GiB": 20, "batch_auto_shrink": False,
            "accuracy_gate_before_other_arms": False,
            "future_evaluation": {"executed_by_this_runner": False,
                "scope": "exploratory_regression_on_previously_observed_data",
                "prior12_not_an_automatic_stop_gate": True,
                "independent_final_available": False},
        })
        self.POLICY["resource_plan"]["comparison_seconds"] = None

    @lru_cache(maxsize=4)
    def epoch_order(self, epoch):
        return tuple(sorted(range(self.POLICY["tiles"]), key=lambda i: hashlib.sha256(
            f"{original.NAME}\0{42}\0{epoch}\0{i}".encode()).digest()))

    def batch_indices(self, index):
        if not 0 <= index < self.POLICY["updates"]:
            raise ValueError("Update outside fixed arm")
        epoch, batch = divmod(index, self.POLICY["updates_per_epoch"])
        return self.epoch_order(epoch)[batch*8:(batch+1)*8]

    def learning_rates(self, index):
        if not 0 <= index < self.POLICY["updates"]:
            raise ValueError("Learning rate index outside fixed arm")
        drops = sum(index >= n for n in self.POLICY["lr_drop_updates"])
        return [rate*.4**drops for rate in (1e-4, 2e-4)]

    def check_disk(self, root, step):
        # All checkpoints are retained. Infer remaining disk need from an actual
        # optimizer-bearing checkpoint, with headroom for larger loss histories.
        files = list(root.glob("step-*/model.pt"))
        size = max((f.stat().st_size for f in files), default=0)
        total, interval = self.POLICY["updates"], self.POLICY["checkpoint_interval"]
        future = [n for n in sorted(set(range(0, total+1, interval)) | {total}) if n >= step]
        needed = max(5*1024**3, int(size*1.3*len(future))+5*1024**3)
        if shutil.disk_usage(root).free < needed:
            raise RuntimeError(f"Insufficient disk to preserve checkpoints: need {needed/1024**3:.1f} GiB free; no deletion")


def prepare(project, handoffs):
    records = read_source(handoffs/"mpi-supersimplenet-fit-v1.zip", ("plan.json", "report.json"))
    source, report = records["plan.json"], records["report.json"]
    if (report["completed_updates"] != 28800 or not report["normal_reload_passed"]
            or source["archive_sha256"] != fit.EXPECTED_ARCHIVE_SHA256
            or report["initial_fingerprints"] != source["expected_fresh_fingerprints"]):
        raise ValueError("Original formal fit provenance differs")
    pool = read_source(handoffs/"mpi-normal-diversity-preparation-v1.zip", ("pool.json",))["pool.json"]
    expanded = select_nested(pool, source["cohorts"])
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project/"scripts/run_mpi_supersimplenet_scaling.ps1",
                    project/"docs/mpi-supersimplenet-scaling-v1.md"])
    return {"name": NAME, "sources": SOURCES, "archive_sha256": source["archive_sha256"],
        "pool_ids_sha256": POOL_IDS_SHA, "pool_rows_sha256": POOL_ROWS_SHA,
        "selection": "original64 in original order plus first192 SHA256(name,NUL,additional-normal,NUL,photo_id)",
        "expanded_training": expanded, "expanded_training_sha256": fit.digest(expanded),
        "original_cohorts": source["cohorts"], "environment": source["environment"],
        "expected_fresh_fingerprints": source["expected_fresh_fingerprints"], "weights": source["weights"],
        "arm_policies": {arm: Protocol(arm).POLICY for arm in ARMS},
        "sources_lf_sha256": {f.relative_to(project).as_posix(): hashlib.sha256(
            f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in paths},
        "test_images_decoded": 0, "test_masks_decoded": 0,
        "note": "Reviewed pool excludes development128/calibration169/validation256 and known test RGB. No new test reads. Exact pixel exclusion does not prove physical-board independence."}


def arm_identity(plan, arm):
    protocol = Protocol(arm)
    groups = deepcopy(plan["original_cohorts"])
    if ARMS[arm][0] == 256:
        groups["training"] = plan["expanded_training"]
    return {"name": NAME+"-"+arm, "scaling_plan_sha256": fit.digest(plan),
        "policy": protocol.POLICY, "cohorts": groups,
        **{k: plan[k] for k in ("environment", "expected_fresh_fingerprints", "weights",
                               "archive_sha256", "sources_lf_sha256")}}
