"""Fail-closed history migration, and immutable bundle integration without GPU."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from defectops import history_provenance as hp
from defectops.manifest import canonical
from defectops.padim_inspection import validate_inference_bundles
from defectops.patchcore_evaluation import load_patchcore_evaluation_config
from defectops.patchcore_experiment import load_patchcore_experiment

ROOT = Path(__file__).resolve().parents[1]
FIT_COMMIT = "e91ecac65d16eabe9d35bb3a5217571e54bd2938"
EVAL_COMMIT = "0946b07b9fed57f18dbe2d12127ed02c64e741c1"


@pytest.fixture
def migrated_repo():
    # Source exports and shallow CI checkouts cannot exercise historical ancestry.
    if hp._git(ROOT, "cat-file", "-e", hp._REWRITTEN_TIP).returncode:
        pytest.skip("Full sanitized Git history required for migration integration")
    return ROOT


def test_manifest_is_pinned_and_contains_only_expected_schema():
    raw = hp._MIGRATION_FILE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hp._MIGRATION_SHA256
    data = json.loads(raw)
    assert len(data["commits"]) == 85
    assert data["policy"] == "entire_git_tree_identical"
    assert FIT_COMMIT in data["commits"] and EVAL_COMMIT in data["commits"]


def test_recorded_fit_resolves_without_old_objects(migrated_repo):
    assert hp._git(ROOT, "merge-base", "--is-ancestor", FIT_COMMIT, "HEAD").returncode != 0
    result = hp.resolve_commit_ancestry(ROOT, FIT_COMMIT, "Fit")
    assert result["method"] == "audited_identical_tree_migration"
    assert result["original_commit"] == FIT_COMMIT
    assert hp._ancestor(ROOT, result["resolved_commit"])


def test_every_approved_mapping_matches_present_history(migrated_repo):
    for original, entry in json.loads(hp._MIGRATION_FILE.read_bytes())["commits"].items():
        result = hp.resolve_commit_ancestry(ROOT, original, "Fit")
        assert result["resolved_commit"] == entry["rewritten_commit"]
        assert result["source_tree"] == entry["tree"]


def test_git_replace_cannot_manufacture_ancestry(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args]).decode().strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    tree = git("write-tree")
    first = git("commit-tree", tree, "-m", "Independent root one")
    second = git("commit-tree", tree, "-m", "Independent root two")
    git("update-ref", "refs/heads/trial", second)
    git("symbolic-ref", "HEAD", "refs/heads/trial")
    git("replace", "--graft", second, first)
    git("merge-base", "--is-ancestor", first, "HEAD")  # Ordinary Git is affected.
    with pytest.raises(RuntimeError, match="no audited"):
        hp.resolve_commit_ancestry(tmp_path, first, "Fit")


def test_direct_ancestry_does_not_depend_on_optional_migration(migrated_repo, tmp_path):
    with patch.object(hp, "_MIGRATION_FILE", tmp_path / "missing"):
        result = hp.resolve_commit_ancestry(ROOT, hp._REWRITTEN_TIP, "Fit")
    assert result["method"] == "direct_ancestor"


@pytest.mark.parametrize("value", [None, "HEAD", "a" * 39, "g" * 40, "--all", "A" * 40])
def test_nonliteral_commit_ids_are_rejected(value):
    with pytest.raises(ValueError, match="full Git commit"):
        hp.resolve_commit_ancestry(ROOT, value, "Fit")


@pytest.mark.parametrize("value", ["f" * 40, "995926e" + "0" * 33])
def test_unknown_commit_is_not_mapped(migrated_repo, value):
    with pytest.raises(RuntimeError, match="no audited"):
        hp.resolve_commit_ancestry(ROOT, value, "Fit")


def test_changed_tree_commit_is_not_whitelisted(migrated_repo):
    # This commit introduced a personal path; its complete tree was altered.
    with pytest.raises(RuntimeError, match="no audited"):
        hp.resolve_commit_ancestry(ROOT, "995926ecd0209618cbff402d20ce801e9a593957", "Fit")


def test_modified_or_missing_map_is_rejected(migrated_repo, tmp_path):
    path = tmp_path / "map.json"
    value = json.loads(hp._MIGRATION_FILE.read_bytes())
    value["commits"][FIT_COMMIT]["rewritten_commit"] = hp._REWRITTEN_TIP
    path.write_text(json.dumps(value))
    with patch.object(hp, "_MIGRATION_FILE", path):
        with pytest.raises(ValueError, match="checksum"):
            hp.resolve_commit_ancestry(ROOT, FIT_COMMIT, "Fit")
        path.unlink()
        with pytest.raises(RuntimeError, match="unavailable"):
            hp.resolve_commit_ancestry(ROOT, FIT_COMMIT, "Fit")


def test_unrelated_repository_rejected(migrated_repo, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-q",
                    "--allow-empty", "-m", "Unrelated history"], check=True)
    with pytest.raises(RuntimeError, match="anchor"):
        hp.resolve_commit_ancestry(tmp_path, FIT_COMMIT, "Fit")


def test_tree_mismatch_rejected(migrated_repo):
    real_git = hp._git

    def wrong_tree(root, *args):
        result = real_git(root, *args)
        if args[0] == "rev-parse":
            result.stdout = "f" * 40 + "\n"
        return result

    with patch.object(hp, "_git", wrong_tree):
        with pytest.raises(ValueError, match="tree differs"):
            hp.resolve_commit_ancestry(ROOT, FIT_COMMIT, "Fit")


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def frozen_bundle(tmp_path):
    """Synthetic checkpoint bytes: tests bundle validation, NOT torch loading."""
    fit, evaluation = tmp_path / "fit", tmp_path / "evaluation"
    fit.mkdir(); evaluation.mkdir()
    manifest = {"protocol": "A-normal-only-v1", "split_seed": 42,
                "categories": ["transistor"], "records": []}
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    for path in (fit, evaluation):
        write_json(path / "manifest.json", manifest)
    for name, folder in [("patchcore-pilot.toml", fit), ("patchcore-evaluation-v1.toml", evaluation)]:
        shutil.copyfile(ROOT / "configs" / name, folder / name)
    config = load_patchcore_experiment(fit / "patchcore-pilot.toml", manifest, 42)
    checkpoint = fit / "checkpoint.bin"
    checkpoint.write_bytes(b"SYNTHETIC CHECKPOINT; NO GPU MODEL")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    status = config["comparison_status"]
    run = {"run_id": "synthetic-fit", "git": {"commit": FIT_COMMIT},
           "config": config, "status": "fit_calibrated_exploratory", "test_evaluated": False,
           "comparison_status": status, "manifest_sha256": manifest["manifest_sha256"],
           "calibration": {"threshold": 40.0},
           "artifacts": {"checkpoint": checkpoint.name, "checkpoint_sha256": sha}}
    write_json(fit / "run.json", run)
    write_json(evaluation / "fit-run.json", run)
    contract = {"status": "exploratory_rules_frozen_before_test_images_loaded",
                "comparison_status": status, "fit_run_id": run["run_id"],
                "image": {"threshold": 40.0},
                "pixel": {"threshold": 10.0, "quantile_method": "higher",
                          "decision_rule": "score >= threshold", "morphology": "none",
                          "connectivity": 4, "minimum_region_area_pixels": 1}}
    path = evaluation / "calibration.json"
    write_json(path, contract)
    value = {"status": "evaluated_exploratory", "test_evaluated": True,
             "evaluation_id": evaluation.name, "comparison_status": status,
             "evaluator_git": {"commit": EVAL_COMMIT},
             "manifest_sha256": manifest["manifest_sha256"],
             "fit": {"run_id": run["run_id"], "git_commit": FIT_COMMIT, "checkpoint_sha256": sha},
             "evaluation_config": load_patchcore_evaluation_config(evaluation / "patchcore-evaluation-v1.toml"),
             "calibration_contract": contract,
             "artifacts": {"calibration_contract": path.name,
                           "calibration_contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    write_json(evaluation / "evaluation.json", value)
    return fit, evaluation


def snapshot(folders):
    return {str(p): p.read_bytes() for folder in folders for p in folder.iterdir()}


def test_complete_bundle_validation_preserves_all_artifact_bytes(migrated_repo, frozen_bundle):
    before = snapshot(frozen_bundle)
    provenance = {}
    run, _, contract, _ = validate_inference_bundles(*frozen_bundle, ROOT, provenance=provenance)
    assert run["git"]["commit"] == FIT_COMMIT
    assert contract["image"]["threshold"] == 40.0
    assert provenance["fit"]["original_commit"] == FIT_COMMIT
    assert provenance["evaluation"]["original_commit"] == EVAL_COMMIT
    assert provenance["fit"]["method"] == "audited_identical_tree_migration"
    assert snapshot(frozen_bundle) == before


@pytest.mark.parametrize("folder,name,error", [
    (0, "checkpoint.bin", "Checkpoint checksum"),
    (1, "calibration.json", "Calibration contract checksum"),
    (1, "fit-run.json", "fit-run snapshot"),
])
def test_artifact_integrity_remains_enforced(migrated_repo, frozen_bundle, folder, name, error):
    path = frozen_bundle[folder] / name
    if name == "fit-run.json":
        value = json.loads(path.read_bytes()); value["run_id"] = "wrong-fit"
        write_json(path, value)
    else:
        path.write_bytes(path.read_bytes() + b"\nchanged")
    before = snapshot(frozen_bundle)
    provenance = {}
    with pytest.raises(ValueError, match=error):
        validate_inference_bundles(*frozen_bundle, ROOT, provenance=provenance)
    assert provenance == {}
    assert snapshot(frozen_bundle) == before


def test_evaluation_without_approved_ancestry_still_fails(migrated_repo, frozen_bundle):
    path = frozen_bundle[1] / "evaluation.json"
    value = json.loads(path.read_bytes()); value["evaluator_git"]["commit"] = "f" * 40
    write_json(path, value)
    with pytest.raises(RuntimeError, match="Evaluation commit has no audited"):
        validate_inference_bundles(*frozen_bundle, ROOT)
