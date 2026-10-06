"""CPU contract/transaction tests; these do not establish CUDA training success."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from scripts import mpi_supersimplenet_scaling_protocol as p
from scripts import run_mpi_supersimplenet_scaling as r
from scripts.mpi_supersimplenet_scaling_runtime import Runtime, Monitor
from scripts import run_mpi_supersimplenet_fit as engine
from tests.test_mpi_supersimplenet_fit import fixture, FakeRuntime, FakeMonitor


def group(ids):
    rows = []
    for pid in ids:
        for region in p.fit.REGIONS:
            key = f'{region}/{pid}'
            rows.append({'photo_id':pid,'region':region,
                'member':f'extracted/{region}/train/good/{pid}.png','width':256,'height':256,
                'member_sha256':hashlib.sha256(key.encode()).hexdigest(),
                'decoded_rgb_sha256':hashlib.sha256(('rgb/'+key).encode()).hexdigest()})
    return {'photo_ids':ids,'rows':rows}


@pytest.fixture
def inputs(monkeypatch):
    g = group([f'{i:04d}' for i in range(1086)])
    pool = {'ids':g['photo_ids'],'rows':g['rows']}
    cohorts = {'training':group(g['photo_ids'][:64]),
               'development':group([f'{i:04d}' for i in range(1100,1228)]),
               'calibration':group([f'{i:04d}' for i in range(1300,1469)])}
    monkeypatch.setattr(p,'POOL_IDS_SHA',p.fit.digest(pool['ids']))
    monkeypatch.setattr(p,'POOL_ROWS_SHA',p.fit.digest(pool['rows']))
    return pool,cohorts


def test_nested256_deterministic_no_source_mutation(inputs):
    pool,cohorts = inputs
    before = p.fit.digest(inputs)
    chosen = p.select_nested(pool,cohorts)
    assert chosen == p.select_nested(pool,cohorts)
    assert chosen['photo_ids'][:64] == cohorts['training']['photo_ids']
    assert chosen['photo_ids'][64:] == sorted(pool['ids'][64:],key=p.rank)[:192]
    assert len(chosen['rows']) == 3072 and p.fit.digest(inputs) == before


def test_pool_tamper_and_protected_overlap_rejected(inputs,monkeypatch):
    pool,cohorts = inputs
    changed = copy.deepcopy(pool); changed['rows'][0]['member'] = 'test/defect/0000.png'
    with pytest.raises(ValueError,match='pool differs'): p.select_nested(changed,cohorts)
    protected = cohorts['development']['rows'][0]['decoded_rgb_sha256']
    pool['rows'][0]['decoded_rgb_sha256'] = protected
    cohorts['training']['rows'][0]['decoded_rgb_sha256'] = protected
    monkeypatch.setattr(p,'POOL_ROWS_SHA',p.fit.digest(pool['rows']))
    with pytest.raises(ValueError): p.select_nested(pool,cohorts)


def test_schedules_and_original_baseline_order():
    for arm,(photos,epochs) in p.ARMS.items():
        x = p.Protocol(arm); policy = x.POLICY
        per = photos*12//8
        assert policy['updates'] == per*epochs
        assert policy['resource_plan']['comparison_seconds'] is None
        assert policy['accuracy_gate_before_other_arms'] is False
        for epoch in (0,epochs-1):
            rows = [i for step in range(epoch*per,(epoch+1)*per) for i in x.batch_indices(step)]
            assert sorted(rows) == list(range(photos*12))
        lo,hi = policy['lr_drop_updates']
        assert x.learning_rates(lo-1) == [1e-4,2e-4]
        assert x.learning_rates(lo) == pytest.approx([4e-5,8e-5])
        assert x.learning_rates(hi) == pytest.approx([1.6e-5,3.2e-5])
        for bad in (-1,policy['updates']):
            with pytest.raises(ValueError): x.batch_indices(bad)
            with pytest.raises(ValueError): x.learning_rates(bad)
    a = p.Protocol('a64-e300')
    for i in (0,95,96,23040,28799):
        assert a.batch_indices(i) == p.original.batch_indices(i)
        assert a.learning_rates(i) == p.original.learning_rates(i)
    assert p.Protocol('c256-e75').learning_rates(23040) != p.Protocol('d256-e300').learning_rates(23040)


def test_checkpoint_final_partial_interval_and_none_time_cap(fixture):
    root,archive,identity,weights,handoff,_ = fixture
    identity['policy'].update(updates=5,epochs=5,updates_per_epoch=1,checkpoint_interval=2,maximum_attempted_updates=8)
    identity['policy']['resource_plan']['comparison_seconds'] = None
    report = engine.execute(root,archive,identity,weights,handoff,runtime_factory=FakeRuntime,monitor_factory=FakeMonitor)
    assert report['completed_updates'] == 5 and report['completed_epochs'] == 5
    assert engine.checkpoint_steps(root,p.fit.digest(identity),identity['policy']) == [0,2,4,5]
    assert engine.execute(root,archive,identity,weights,handoff,runtime_factory=FakeRuntime,monitor_factory=FakeMonitor) == report


def test_injected_protocol_resume(fixture):
    root,archive,identity,weights,handoff,_ = fixture
    identity['policy']['resource_plan']['comparison_seconds'] = None
    identity['policy']['updates_per_epoch'] = 1
    custom = SimpleNamespace(batch_indices=lambda n:range(8),validate_predictions=p.original.validate_predictions,
        tile_scores=p.original.tile_scores,calibrate=lambda d,c:{'centers':[0.]*12,'scales':[1.]*12,'threshold':6.},
        photo_scores=p.original.photo_scores)
    class Interrupted(FakeRuntime): interrupt=3
    with pytest.raises(KeyboardInterrupt):
        engine.execute(root,archive,identity,weights,handoff,runtime_factory=Interrupted,monitor_factory=FakeMonitor,protocol=custom)
    report = engine.execute(root,archive,identity,weights,handoff,runtime_factory=FakeRuntime,monitor_factory=FakeMonitor,protocol=custom)
    assert report['completed_updates'] == 4 and report['cumulative_attempted_updates'] == 6


def test_memory_still_enforced_without_clock_limit():
    m = Monitor.__new__(Monitor)
    m.runtime=None; m.sample=lambda:None; m.error=None; m.rss=0; m.reserved=0
    m.limits={'rss_GiB':20,'selection_and_inference_reserved_GiB':6,'comparison_seconds':None}
    m.check()
    m.reserved=7*1024**3
    with pytest.raises(RuntimeError,match='memory limit'): m.check()


def test_runtime_keeps_public_step_and_lrs(monkeypatch):
    from scripts.mpi_supersimplenet_runtime import Runtime as Base
    x=Runtime.__new__(Runtime); x.protocol=p.Protocol('d256-e300')
    x.optimizer=SimpleNamespace(param_groups=[{},{}])
    monkeypatch.setattr(Base,'step',lambda self,images,index:{'official_base_called':True})
    row=x.step(['sentinel'],92160)
    assert row['official_base_called'] and row['epoch']==241
    assert [g['lr'] for g in x.optimizer.param_groups] == pytest.approx([4e-5,8e-5])


def test_pack_idempotence_and_tamper(tmp_path):
    path=tmp_path/'x.zip'
    r.pack(path,{'report.json':{'test_images_decoded':0}})
    before=path.read_bytes(); r.pack(path,{'report.json':{'test_images_decoded':0}})
    assert path.read_bytes()==before
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert manifest['report.json']==hashlib.sha256(z.read('report.json')).hexdigest()
    with pytest.raises(ValueError): r.pack(path,{'report.json':{'changed':True}})


def test_reviewed_windows_sources_when_available():
    folder=Path.home()/'Downloads'
    if not all((folder/n).exists() for n in p.SOURCES): pytest.skip('Reviewed pool ZIP retained on Windows, not present locally')
    plan=p.prepare(r.ROOT,folder)
    assert len(plan['expanded_training']['photo_ids'])==256
    for arm in p.ARMS:
        assert p.arm_identity(plan,arm)['expected_fresh_fingerprints']==plan['expected_fresh_fingerprints']


def test_prepare_from_reviewed_schema_pins_every_arm(inputs,tmp_path,monkeypatch):
    pool,cohorts=inputs
    fp={'backbone':'fixed','adaptor':'fresh','segmentation':'fresh','classification':'fresh'}
    source={'cohorts':cohorts,'archive_sha256':p.fit.EXPECTED_ARCHIVE_SHA256,
            'expected_fresh_fingerprints':fp,'environment':{'fake_not_cuda':True},'weights':{}}
    report={'completed_updates':28800,'normal_reload_passed':True,'initial_fingerprints':fp}
    monkeypatch.setattr(p,'read_source',lambda path,names:
        {'plan.json':source,'report.json':report} if path.name=='mpi-supersimplenet-fit-v1.zip' else {'pool.json':pool})
    plan=p.prepare(r.ROOT,tmp_path)
    for arm in p.ARMS:
        identity=p.arm_identity(plan,arm)
        assert identity['scaling_plan_sha256']==p.fit.digest(plan)
        assert len(identity['cohorts']['training']['photo_ids'])==p.ARMS[arm][0]
        assert identity['cohorts']['development']==cohorts['development']
        assert identity['expected_fresh_fingerprints']==fp
    assert plan['expanded_training_sha256']==p.fit.digest(plan['expanded_training'])


def test_normal_preflight_resumes_without_test_reads(inputs,tmp_path,monkeypatch):
    pool,cohorts=inputs
    plan={'expanded_training':p.select_nested(pool,cohorts),'original_cohorts':cohorts}
    archive=tmp_path/'source.zip'
    all_rows=plan['expanded_training']['rows']+cohorts['development']['rows']+cohorts['calibration']['rows']
    with zipfile.ZipFile(archive,'w') as z:
        for row in all_rows:z.writestr(row['member'],b'fake')
        z.writestr('extracted/test/defect/never.png',b'forbidden')
    from PIL import Image
    seen=[]
    def check(source,row):
        assert '/train/good/' in row['member']
        seen.append(row['member'])
        return Image.new('RGB',(256,256))
    monkeypatch.setattr(r.fit,'checked_image',check)
    root=tmp_path/'run'; root.mkdir()
    r.publish(root,'identity',p.fit.digest(plan),plan)
    summary,receipts=r.normal_preflight(root,archive,plan)
    assert len(seen)==6636 and len(receipts)==553
    assert summary['test_images_decoded']==0
    assert r.normal_preflight(root,archive,plan)==(summary,receipts) and len(seen)==6636
    r.verify_prepared(root,plan)
    first=root/'normal-integrity'/f"training256-{plan['expanded_training']['photo_ids'][0]}"/'value.json'
    first.write_bytes(b'corrupt')
    with pytest.raises(ValueError):r.verify_prepared(root,plan)


def test_disk_failure_does_not_delete_checkpoints(tmp_path,monkeypatch):
    root=tmp_path/'step-000000';root.mkdir()
    model=root/'model.pt';model.write_bytes(b'preserved')
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    with pytest.raises(RuntimeError,match='Insufficient disk'):p.Protocol('c256-e75').check_disk(tmp_path,3840)
    assert model.read_bytes()==b'preserved'
