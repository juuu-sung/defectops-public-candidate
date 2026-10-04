"""CPU orchestration/regression tests, not a formal CUDA training run."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_supersimplenet_fit_protocol as p
from scripts import mpi_supersimplenet_fit_runtime as rt
from scripts import run_mpi_supersimplenet_fit as r
from tests.test_mpi_efficientad_fit import cohorts


def test_epoch_schedule_every_tile_once_and_lr_boundaries():
    assert p.POLICY['updates'] == 300*96 == 28800
    assert p.POLICY['checkpoint_interval'] == 960
    assert p.POLICY['selected_checkpoint'] == 'epoch300_step28800_only'
    for epoch in (0, 1, 239, 240, 269, 270, 299):
        values = [i for step in range(epoch*96, (epoch+1)*96) for i in p.batch_indices(step)]
        assert sorted(values) == list(range(768))
    assert p.epoch_order(0) != p.epoch_order(1)
    assert p.learning_rates(23039) == [1e-4, 2e-4]
    assert p.learning_rates(23040) == pytest.approx([4e-5, 8e-5])
    assert p.learning_rates(25919) == pytest.approx([4e-5, 8e-5])
    assert p.learning_rates(25920) == pytest.approx([1.6e-5, 3.2e-5])
    assert p.learning_rates(28799) == pytest.approx([1.6e-5, 3.2e-5])
    for bad in (-1, 28800):
        with pytest.raises(ValueError): p.batch_indices(bad)
        with pytest.raises(ValueError): p.learning_rates(bad)


def test_native_logit_not_heatmap_is_primary_score():
    values = np.full((12, 65537), 100, dtype=np.float32)
    values[:, 0] = -2
    assert np.array_equal(p.tile_scores(values), np.full(12, -2.))
    for bad in (values.astype(np.float64), values[:, 1:], values*np.nan):
        with pytest.raises(ValueError): p.validate_predictions(bad)


def test_signed_median_q90_and_higher_q95_contract():
    dev = np.arange(128)[:, None] + np.arange(12)[None, :] - 150.
    cal = np.arange(169)[:, None] + np.arange(12)[None, :] - 150.
    contract = p.calibrate(dev, cal)
    assert np.array_equal(contract['centers'], np.median(dev, axis=0))
    assert np.array_equal(contract['scales'], np.quantile(dev, .9, axis=0, method='higher')-np.median(dev, axis=0))
    expected = sorted(p.photo_scores(cal, contract))[int(np.ceil(.95*168))]
    assert expected == contract['threshold']
    assert contract['test_scores_used'] == 0 and contract['normalization_clipped'] is False
    assert p.photo_scores(np.full(12, -1000), contract) < 0
    for bad in (dev[:-1], dev*np.nan, np.zeros((128, 12)), np.full((128, 12), 1e-7)):
        with pytest.raises(ValueError): p.calibrate(bad, cal)
    with pytest.raises(ValueError): p.photo_scores(cal, {**contract, 'scales': [0.]*12})


def test_cohorts_disjoint_and_no_defect_paths():
    c = cohorts()
    p.validate_cohorts(c)
    c['development']['rows'][0]['member'] = 'test/defect/never.png'
    with pytest.raises(ValueError): p.validate_cohorts(c)


def test_preparation_reads_reviewed_handoffs_only():
    handoffs = Path.home()/'Downloads'
    required = ['mpi-supersimplenet-probe-v1.zip', 'mpi-pcb-normal-bank-preparation-v1.zip', 'mpi-pcb-score-probe-v1.zip']
    if not all((handoffs/n).exists() for n in required): pytest.skip('Local reviewed handoffs unavailable')
    identity = p.prepare(r.ROOT, handoffs)
    assert [len(identity['cohorts'][g]['photo_ids']) for g in ('training','development','calibration')] == [64,128,169]
    assert identity['environment']['gpu'] == 'NVIDIA GeForce RTX 5060'
    assert identity['policy']['future_evaluation']['executed_by_this_runner'] is False


class FakeRuntime:
    interrupt = None
    drift = False
    def __init__(self, *args): self.value = 0; self.loaded = False
    def fingerprints(self):
        return {'backbone':'fixed', 'adaptor':str(self.value), 'segmentation':str(self.value), 'classification':str(self.value)}
    def step(self, images, index):
        assert len(images) == 8
        if index == self.interrupt: raise KeyboardInterrupt()
        self.value = index+1
        return {'step':index+1, 'loss':1., 'synthetic_anomalous_samples':8, 'fake_not_cuda':True}
    def predict(self, image):
        n = image.getpixel((0,0))[0]
        return np.full(65537, n+self.value+(1 if self.loaded and self.drift else 0), np.float32)
    def save(self, path, signature, step):
        path.write_text(json.dumps({'identity':signature, 'step':step}))
    def load(self, path, signature, step):
        assert json.loads(path.read_text()) == {'identity':signature, 'step':step}
        self.value = step; self.loaded = True


class FakeMonitor:
    def __init__(self, *args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {'fake_not_cuda':True}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    policy = copy.deepcopy(p.POLICY)
    policy.update(updates=4, checkpoint_interval=2, maximum_attempted_updates=6)
    groups = {}; members = []
    for g,n in (('training',1),('development',2),('calibration',3)):
        ids = [f'{i:04d}' for i in range(n)]
        rows = [{'member':f'train/good/{g}/{i}/{j}', 'photo_id':ids[i], 'region':j}
                for i in range(n) for j in range(12)]
        groups[g] = {'photo_ids':ids, 'rows':rows}; members += rows
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for row in members: z.writestr(row['member'],b'fake')
        z.writestr('test/defect/never',b'forbidden')
    seen = []
    def checked(z,row):
        assert row['member'].startswith('train/good/')
        seen.append(row['member'])
        return Image.new('RGB',(256,256),(int(row['member'].split('/')[-2])+1,0,0))
    monkeypatch.setattr(r.fit,'checked_image',checked)
    monkeypatch.setattr(p,'batch_indices',lambda i:list(range(8)))
    monkeypatch.setattr(p,'calibrate',lambda d,c:{'centers':[0.]*12,'scales':[1.]*12,'threshold':6.})
    identity = {'policy':policy,'cohorts':groups,'environment':{'fake':True},'expected_fresh_fingerprints':FakeRuntime().fingerprints()}
    return tmp_path/'run', archive, identity, tmp_path/'weights', tmp_path/'handoff.zip', seen


def execute(fixture, runtime=FakeRuntime, monitor=FakeMonitor):
    return r.execute(*fixture[:5],runtime_factory=runtime,monitor_factory=monitor)


def test_complete_reload_json_export_idempotent(fixture):
    report = execute(fixture)
    assert report['completed_updates'] == 4 and report['test_images_decoded'] == 0
    assert report['accuracy_measured'] is False
    count = len(fixture[-1])
    assert execute(fixture) == report and len(fixture[-1]) == count
    with zipfile.ZipFile(fixture[4]) as z:
        assert all(n.endswith('.json') for n in z.namelist())
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert all(r.hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())
        assert json.loads(z.read('reload.json'))['decision_changes']==0


def test_ctrl_c_resume_counts_lost_work(fixture):
    class Interrupted(FakeRuntime): interrupt=3
    with pytest.raises(KeyboardInterrupt): execute(fixture,Interrupted)
    assert (fixture[0]/'step-000002/model.pt').exists()
    report=execute(fixture)
    assert report['completed_updates']==4 and report['cumulative_attempted_updates']==6


def test_resource_failure_blocks_retry(fixture):
    class Failing(FakeMonitor):
        def check(self): raise RuntimeError('resource cap')
    with pytest.raises(RuntimeError): execute(fixture,monitor=Failing)
    with pytest.raises(ValueError,match='review'): execute(fixture)


def test_reload_drift_stops_before_export(fixture):
    class Drifting(FakeRuntime): drift=True
    with pytest.raises(ValueError,match='drift'): execute(fixture,Drifting)
    assert (fixture[0]/'step-000004/model.pt').exists() and not fixture[4].exists()


def test_wrong_fresh_initialization_blocks_training(fixture):
    fixture[2]['expected_fresh_fingerprints']['adaptor']='wrong'
    with pytest.raises(ValueError,match='Fresh model'): execute(fixture)
    assert not (fixture[0]/'step-000000').exists()


def test_changed_identity_and_raw_cache_refused(fixture):
    execute(fixture)
    fixture[2]['changed']=True
    with pytest.raises(ValueError): execute(fixture)
    del fixture[2]['changed']
    (fixture[0]/'development/photo-000/predictions.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='incomplete or changed'): execute(fixture)


def test_forced_kill_and_checkpoint_gap_refused(fixture):
    execute(fixture)
    (fixture[0]/'attempt-forced').mkdir()
    with pytest.raises(ValueError,match='Unfinished'): execute(fixture)


def test_checkpoint_gap_refused(fixture):
    execute(fixture)
    (fixture[0]/'step-000002').rename(fixture[0]/'.kept-two')
    with pytest.raises(ValueError,match='gap'): execute(fixture)


def test_resume_budget_not_reset(fixture):
    class Interrupted(FakeRuntime): interrupt=1
    fixture[2]['policy']['maximum_attempted_updates']=4
    with pytest.raises(KeyboardInterrupt): execute(fixture,Interrupted)
    with pytest.raises(RuntimeError,match='attempted-update'): execute(fixture)
    assert not fixture[4].exists()


def test_time_cap_before_cuda(fixture):
    fixture[2]['policy']['resource_plan']['comparison_seconds']=0
    with pytest.raises(RuntimeError,match='time cap'): execute(fixture)
    assert not list(fixture[0].glob('attempt-*'))


def test_formal_checkpoint_separate_purpose_and_cli(capsys):
    from scripts.mpi_supersimplenet_runtime import Runtime as Probe
    assert rt.Runtime.CHECKPOINT_PURPOSE == p.NAME != Probe.CHECKPOINT_PURPOSE
    with pytest.raises(SystemExit) as e: r.main(['--help'])
    assert e.value.code==0 and '--epochs' not in capsys.readouterr().out


def test_probe_checkpoint_cannot_be_loaded_by_formal_runtime():
    from scripts.mpi_supersimplenet_runtime import Runtime as Probe
    runtime=rt.Runtime.__new__(rt.Runtime)
    runtime.torch=SimpleNamespace(load=lambda *a,**k:{'purpose':Probe.CHECKPOINT_PURPOSE,'identity':'sig','step':0})
    with pytest.raises(ValueError,match='identity differs'): runtime.load(Path('unused'), 'sig', 0)


def test_actual_runtime_applies_both_learning_rates(monkeypatch):
    from scripts.mpi_supersimplenet_runtime import Runtime as Probe
    monkeypatch.setattr(Probe,'step',lambda self,images,index:{'step':index+1})
    runtime=rt.Runtime.__new__(rt.Runtime)
    runtime.optimizer=SimpleNamespace(param_groups=[{},{}])
    result=runtime.step([],25920)
    assert [g['lr'] for g in runtime.optimizer.param_groups]==pytest.approx([1.6e-5,3.2e-5])
    assert result['epoch']==271


def test_final_normal_only_map_drift_is_rejected(fixture):
    class Drift(FakeRuntime):
        loads=0
        def load(self,*args):
            super().load(*args); self.loads+=1
        def predict(self,image):
            values=super().predict(image)
            if self.loads>=2 and image.getpixel((0,0))[0]>1: values[100]+=1
            return values
    with pytest.raises(ValueError,match='Final normal reload drift'): execute(fixture,Drift)
    assert (fixture[0]/'calibrated').exists() and not fixture[4].exists()


def test_handoff_tampering_not_overwritten(fixture):
    execute(fixture)
    with zipfile.ZipFile(fixture[4],'w') as z: z.writestr('wrong.json',b'{}')
    before=fixture[4].read_bytes()
    with pytest.raises(ValueError,match='handoff differs'): execute(fixture)
    assert fixture[4].read_bytes()==before
