"""CPU orchestration evidence only; synthetic inference is not GPU validation."""
import copy
import hashlib
import json
import zipfile

import pytest

from scripts import mpi_score_validation_protocol as p
from scripts import validate_mpi_scores as runner
from tests.test_mpi_score_probe import experiment, fingerprint, scored
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, Runtime
from tests.test_mpi_pcb_normal_bank_fit import png_row


def test_contract_uses_separate_normals_and_exact_higher_quantiles():
    dev = [scored("development",str(i),v,raw=2*v) for i,v in enumerate((0,1,2,3,20))]
    cal = [scored("calibration",str(i+10),v,raw=2*v) for i,v in enumerate((1,2,5,8))]
    c = p.fit_contract(dev,cal)
    q = c["candidate_top1_q90"]
    assert q["regions"][0] == {"region":p.fit.REGIONS[0],"center":2.,"scale":18.}
    assert q["threshold"] == pytest.approx(1/3)
    assert p.score(cal[-1],q)["alert"] is True  # Exact threshold is >=, not >.
    assert c["control_top1_mad"]["regions"][0]["scale"] == 1
    with pytest.raises(ValueError,match="designated normal"):
        p.fit_contract(dev,[dict(cal[0],label=1)])
    with pytest.raises(ValueError,match="overlap"):
        p.fit_contract(dev,[scored("calibration","0",1)])
    with pytest.raises(ValueError,match="Degenerate"):
        p.fit_contract([dev[0]],cal)


def test_unused_selection_excludes_ids_pixels_and_is_input_order_independent():
    rows = [fingerprint(f"{i:04d}",r) for i in range(15) for r in p.fit.REGIONS]
    prep = {"training_photo_ids":["0000"],"calibration_photo_ids":["0001"],
            "training_rows":rows[:12],"calibration_rows":rows[12:24]}
    dev = {"photo_ids":["0002"],"rows":rows[24:36]}
    audit = [{"photo_id":"0003","regions":rows[36:48]}]
    rows[48]["decoded_rgb_sha256"] = rows[24]["decoded_rgb_sha256"]
    selected = p.select_normals(rows,prep,dev,audit,count=8)
    assert not set(selected["photo_ids"]) & {"0000","0001","0002","0003","0004"}
    assert selected == p.select_normals(list(reversed(rows)),prep,dev,audit,count=8)
    with pytest.raises(ValueError,match="Insufficient"):
        p.select_normals(rows,prep,dev,audit,count=11)
    with pytest.raises(ValueError,match="Duplicate"):
        p.select_normals(rows+rows[:1],prep,dev,audit,count=1)
    with pytest.raises(ValueError,match="Incomplete"):
        p.select_normals(rows[:-1],prep,dev,audit,count=1)


def test_replay_checks_native_and_small_decision_flips():
    dev = [scored("development",str(i),v) for i,v in enumerate((1,2,3))]
    cal = [scored("calibration","10",4)]
    c = p.fit_contract(dev,cal)
    changed = copy.deepcopy(cal[0])
    changed["regions"][0]["native_raw"] += .01
    with pytest.raises(ValueError,match="Raw score"):
        p.replay(changed,cal[0],c)
    changed = copy.deepcopy(cal[0])
    for row in changed["regions"]:
        row["top1"] -= 1e-7
    with pytest.raises(ValueError,match="decision changed"):
        p.replay(changed,cal[0],c)


@pytest.fixture
def validation_run(experiment):
    archive,args = experiment
    dev = args.pop("development")
    args.pop("baseline")
    args.pop("selection")
    cohorts = {
        "development":runner.probe.normal_photos(dev["photo_ids"],dev["rows"],"development"),
        "calibration":runner.probe.normal_photos(args["preparation"]["calibration_photo_ids"],args["preparation"]["calibration_rows"],"calibration"),
        "test":args["previous"]["test-audit.json"]["photos"],
    }
    values = {"development":[15,16,17],"calibration":[10,11],"test":[10,70,10,70]}
    reference = {g:[{k:s[k] for k in ("key","photo_id","label")} | {"regions":[
        {k:r[k] for k in ("region","member")} | {"top1":float(v),"native_raw":float(v)} for r in s["regions"]]}
        for s,v in zip(photos,values[g],strict=True)] for g,photos in cohorts.items()}
    rows = []
    with zipfile.ZipFile(archive,"a") as z:
        for region in p.fit.REGIONS:
            data,row = png_row("0800",region,9)
            rows.append(row)
            z.writestr(row["member"],data)
    cohorts["validation"] = runner.probe.normal_photos(["0800"],rows,"validation")
    contract = p.fit_contract(reference["development"],reference["calibration"])
    frozen = {"contract":contract,"observed_test_reference":p.observed_metrics(reference["test"],contract)}
    args.update(cohorts=cohorts,reference=reference,frozen=frozen)
    return archive,args


def execute(data,calls,**options):
    path,args = data
    with zipfile.ZipFile(path) as source:
        return runner.execute(source=source,**args,runtime_factory=lambda cfg:Runtime(calls,**options))


def test_complete_gate_manifest_frozen_thresholds_and_no_gpu_rerun(validation_run):
    calls = []
    result = execute(validation_run,calls)
    assert calls.count("predict") == 25 + (3+2+4+1)*12
    assert calls[-1] == "release"
    assert result["replay"]["passed"] and result["replay"]["photos"] == 9
    assert result["normal_validation"]["thresholds_refitted"] is False
    assert result["normal_validation"]["new_defect_photos"] == 0
    for name,method in result["normal_validation"]["methods"].items():
        assert method["false_positives"] == 0
        assert method["threshold"] == validation_run[1]["frozen"]["contract"][name]["threshold"]
    output = validation_run[1]["handoff"]
    with zipfile.ZipFile(output) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == set(manifest) | {"handoff-manifest.json"}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    before = output.read_bytes()
    calls = []
    assert execute(validation_run,calls) == result
    assert calls == [] and output.read_bytes() == before


def test_interruption_reuses_completed_samples_and_rechecks_runtime(validation_run):
    with pytest.raises(RuntimeError,match="interruption"):
        execute(validation_run,[],fail_after=25+12)
    root = validation_run[1]["root"]
    before = (root/"development/photo-001/sample.json").read_bytes()
    assert not (root/"replay").exists() and not list((root/"validation").glob("photo-*"))
    calls = []
    execute(validation_run,calls)
    assert calls.count("predict") == 25 + (2+2+4+1)*12
    assert (root/"development/photo-001/sample.json").read_bytes() == before


def test_replay_failure_stops_before_any_unused_normal_read(validation_run):
    validation_run[1]["reference"]["test"][0]["regions"][0]["native_raw"] += 1
    with pytest.raises(ValueError,match="Raw score replay drift"):
        execute(validation_run,[])
    root = validation_run[1]["root"]
    assert not (root/"replay").exists()
    assert not list((root/"validation").glob("photo-*"))
    assert not validation_run[1]["handoff"].exists()


def test_probe_drift_stops_before_development(validation_run):
    with pytest.raises(ValueError,match="Normal probe blocked"):
        execute(validation_run,[],offset=1)
    assert not list((validation_run[1]["root"]/"development").glob("photo-*"))


def test_cache_or_probe_tampering_stops_without_gpu(validation_run):
    execute(validation_run,[])
    root = validation_run[1]["root"]
    receipt = json.loads((root/"development/photo-001/receipt.json").read_bytes())
    (root/receipt["probe_path"]).write_text("{}")
    calls = []
    with pytest.raises(ValueError,match="probe evidence"):
        execute(validation_run,calls)
    assert calls == []
