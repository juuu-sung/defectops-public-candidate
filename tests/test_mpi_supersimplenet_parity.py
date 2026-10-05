"""CPU protocol/orchestration tests; optional small torch tests are not GPU evidence."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_supersimplenet_parity_protocol as p
from scripts import mpi_supersimplenet_parity_reference as ref
from scripts import run_mpi_supersimplenet_parity as runner


def test_locked_scope_and_jobs():
    assert len(p.jobs()) == 4
    assert {(j['step'],j['seed']) for j in p.jobs()} == {(s,r) for s in (0,28800) for r in (42,900001)}
    assert p.POLICY['later_phases_automatically_started'] is False
    assert p.POLICY['test_images_decoded'] == p.POLICY['test_masks_decoded'] == 0
    assert p.POLICY['maximum_attempted_jobs'] == 5


@pytest.mark.parametrize('kind', list(p.POLICY['tolerances']))
def test_pointwise_tolerance_and_zero_reference(kind):
    tol = p.POLICY['tolerances'][kind][0]
    assert p.compare([0.],[0.],kind)['passed']
    assert p.compare([tol/2],[0.],kind)['passed']
    bad = p.compare([tol*2],[0.],kind)
    assert not bad['passed'] and bad['outside_tolerance'] == 1 and bad['relative_l2'] is None
    # A large unrelated element must not hide an error on a small value.
    assert not p.compare([1e8,tol*2],[1e8,0.],kind)['passed']


@pytest.mark.parametrize('a,b', [([],[]),([np.nan],[0.]),([0.],[np.inf]),([1.],[[1.]]),(['x'],['x'])])
def test_bad_arrays_rejected(a,b):
    with pytest.raises(ValueError): p.compare(a,b,'outputs')


def test_independent_reference_and_cli_import_without_torch():
    source = inspect.getsource(ref)
    assert 'from anomalib' not in source and 'import anomalib' not in source
    assert 'MIT License' in source and p.AUTHOR_COMMIT in source
    run = subprocess.run([sys.executable,str(runner.ROOT/'scripts/run_mpi_supersimplenet_parity.py'),'--help'],capture_output=True,text=True)
    assert run.returncode == 0 and '--archive' in run.stdout


class Monitor:
    def __init__(self,*args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def check(self): pass
    def snapshot(self): return {'fake_not_cuda':True}


class Runtime:
    calls = 0
    interrupt = None
    fail = None
    mismatch = False
    def __init__(self,*args): pass
    def release(self): pass
    def compare_job(self,images,job,*args):
        type(self).calls += 1
        if type(self).calls == self.interrupt: raise KeyboardInterrupt()
        if type(self).calls == self.fail: raise RuntimeError('resource failure')
        assert len(images) == 8
        a = np.zeros((16,64,64),np.float32)
        b = a+(.1 if self.mismatch else 0.)
        arrays = {'official_maps':a,'reference_maps':b,'official_scores':np.zeros(16,np.float32),'reference_scores':np.zeros(16,np.float32)}
        checks = {k:p.compare(arrays['official_'+k],arrays['reference_'+k],'outputs') for k in ('maps','scores')}
        return {'job':job,'model_state_restored':True,'passed':not self.mismatch,'checks':checks,'loss_values':{}},arrays


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    Runtime.calls = 0
    root, fit_root = tmp_path/'audit',tmp_path/'fit'
    checkpoints = {}
    for step in (0,28800):
        path = fit_root/f'step-{step:06d}'/'model.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(f'original-{step}'.encode())
        checkpoints[str(step)] = {'sha256':runner.fit.sha256_file(path),'fingerprints':{}}
    (fit_root/'calibrated').mkdir()
    (fit_root/'calibrated/value.json').write_text(json.dumps({'contract':{'threshold':3}}))
    rows = [{'member':f'extracted/grid{i}/train/good/1073.png'} for i in range(8)]
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for row in rows: z.writestr(row['member'],b'normal')
        z.writestr('test/defect/never.png',b'forbidden')
    seen = []
    def checked(z,row):
        assert '/train/good/' in row['member']
        seen.append(row['member'])
        return Image.new('RGB',(256,256))
    monkeypatch.setattr(runner.fit,'checked_image',checked)
    identity = {'jobs':p.jobs(),'rows':rows,'fit_signature':'fixed','checkpoints':checkpoints,
                'contract_sha256':runner.fit.digest({'threshold':3})}
    return root,archive,fit_root,identity,tmp_path/'result.zip',seen


def execute(fixture,runtime=Runtime):
    return runner.execute(*fixture[:5],runtime_factory=runtime,monitor_factory=Monitor)


def test_complete_preserve_and_idempotent_export(fixture):
    result = execute(fixture)
    assert result['all_matched'] and result['completed_jobs'] == 4
    assert result['completed_disposable_updates'] == 8 and result['saved_updated_models'] == 0
    assert not result['later_phases_automatically_started']
    assert len(fixture[5]) == 32
    assert execute(fixture) == result and len(fixture[5]) == 32
    assert not list(fixture[0].rglob('model.pt'))
    with zipfile.ZipFile(fixture[4]) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())


def test_difference_is_reviewable_result_not_automatic_retry(fixture):
    class Different(Runtime): mismatch = True
    result = execute(fixture,Different)
    assert result['status'] == 'implementation_parity_needs_review'
    assert result['completed_jobs'] == 4 and fixture[4].exists()


def test_interrupt_resume_keeps_completed_job_and_counts_attempts(fixture):
    class Interrupted(Runtime): interrupt = 2
    with pytest.raises(KeyboardInterrupt): execute(fixture,Interrupted)
    result = execute(fixture)
    assert result['attempted_jobs_including_interrupted'] == 5
    assert result['all_matched']


def test_failed_attempt_not_retried(fixture):
    class Failed(Runtime): fail = 1
    with pytest.raises(RuntimeError,match='resource failure'): execute(fixture,Failed)
    with pytest.raises(ValueError,match='Prior failed'): execute(fixture)


def test_checkpoint_and_contract_tampering_block(fixture):
    execute(fixture)
    (fixture[2]/'step-000000/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'): execute(fixture)


def test_contract_change_is_rejected(fixture):
    (fixture[2]/'calibrated/value.json').write_text(json.dumps({'contract':{'threshold':4}}))
    with pytest.raises(ValueError,match='threshold contract changed'): execute(fixture)


def test_duplicate_or_test_member_is_rejected(fixture):
    fixture[3]['rows'][0]['member'] = 'test/defect/never.png'
    with pytest.raises(ValueError,match='non-normal input'): execute(fixture)
    assert not fixture[5]


def test_normal_completed_fit_required_before_gpu(tmp_path):
    assert 'step-000000' in (runner.ROOT/'docs/mpi-supersimplenet-implementation-parity-v1.md').read_text()
    assert 'prepare(' in inspect.getsource(runner.main)
    with pytest.raises(FileNotFoundError):
        p.prepare(tmp_path,tmp_path,tmp_path)


def test_cached_array_tampering_block(fixture):
    execute(fixture)
    path = fixture[0]/'jobs'/p.jobs()[0]['name']/'arrays.npz'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='incomplete or changed'): execute(fixture)


def test_unexpected_output_preserved(fixture):
    fixture[0].mkdir()
    (fixture[0]/'user.txt').write_text('keep')
    with pytest.raises(ValueError,match='without identity'): execute(fixture)
    assert (fixture[0]/'user.txt').read_text() == 'keep'


def test_reference_small_tensor_forward_backward_if_torch_available():
    torch = pytest.importorskip('torch')
    model = ref.build_reference(channels=4,hidden=3,classification_channels=2)
    features = torch.randn(2,4,8,8)
    noise = torch.randn(4,4,8,8)*.015
    maps,scores = model(features,noise)
    masks = torch.zeros_like(maps); masks[0,:,2:5,2:5] = 1
    labels = masks.flatten(1).any(1).float()
    loss,parts = ref.reference_loss(maps,scores,masks,labels)
    loss.backward()
    assert maps.shape == masks.shape and scores.shape == (4,)
    assert len(parts) == 3 and all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
