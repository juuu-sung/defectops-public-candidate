"""CPU synthetic runtime tests; not measured GPU performance or model accuracy."""
import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import compare_mpi_regions as run
from scripts import mpi_region_search_protocol as p
from scripts.mpi_region_runtime import selected_bank, RegionRuntime
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, Runtime
from tests.test_mpi_pcb_normal_bank_fit import png_row


class RegionFake(Runtime):
    def __init__(self,calls,feature_fail_after=None,region_fail_after=None,missing=False,mutate=False,**kwargs):
        super().__init__(calls,**kwargs)
        self.feature_calls=self.region_calls=self.bank_reads=0
        self.feature_fail_after,self.region_fail_after=feature_fail_after,region_fail_after
        self.missing,self.mutate=missing,mutate
    def bank_fingerprint(self):
        self.bank_reads+=1
        return {'shape':[9,768],'float32_le_sha256':'changed' if self.mutate and self.bank_reads>1 else 'synthetic'}
    def verify_feature_path(self,image):
        self.calls.append('feature_gate')
        return {'direct_replay_exact':True}
    def all_bank_hashes(self):
        return p.trace.vector_hashes(np.array([[i]*768 for i in range(1,10)],dtype=np.float32))
    def candidate_features(self,image):
        self.feature_calls+=1;self.calls.append('features')
        if self.feature_fail_after and self.feature_calls>self.feature_fail_after:
            raise RuntimeError('synthetic feature interruption')
        values=np.full((1024,768),10,dtype=np.float32)
        if not self.missing:
            values[:9]=np.array([[i]*768 for i in range(1,10)],dtype=np.float32)
        return values,p.fit.spatial_patch_indices(64,2)
    def predict_region(self,image,indices):
        self.region_calls+=1;self.calls.append('region')
        assert indices==list(range(9))
        if self.region_fail_after and self.region_calls>self.region_fail_after:
            raise RuntimeError('synthetic scoring interruption')
        value=float(image.getpixel((0,0))[0])+2
        return value,np.full((256,256),value,dtype=np.float32)


@pytest.fixture
def experiment(fixture):
    d=fixture
    train_rows,dev_rows=[],[]
    with zipfile.ZipFile(d['source_path'],'a') as z:
        for target,ids,offset in ((train_rows,['0400','0401'],22),(dev_rows,['0200','0201','0202'],15)):
            for i,pid in enumerate(ids):
                for region in p.fit.REGIONS:
                    raw,row=png_row(pid,region,offset+i)
                    z.writestr(row['member'],raw);target.append(row)
    prep=d['preparation'];prep.update(training_photo_ids=['0400','0401'],training_rows=train_rows)
    dev={'photo_ids':['0200','0201','0202'],'rows':dev_rows}
    photos={'development':run.old.normal_photos(dev['photo_ids'],dev_rows,'development'),
            'calibration':run.old.normal_photos(prep['calibration_photo_ids'],prep['calibration_rows'],'calibration'),
            'test':d['previous']['test-audit.json']['photos']}
    raw={}
    for g,ps in photos.items():
        raw[g]=[]
        for photo,value in zip(ps,{'development':[15,16,17],'calibration':[10,11],'test':[10,70,10,70]}[g],strict=True):
            raw[g].append({k:photo[k] for k in ('key','photo_id','label')} | {'regions':[
                {'region':r['region'],'member':r['member'],'top1':float(value),'native_raw':float(value)} for r in photo['regions']]})
    contract=p.calibrate(raw['development'],raw['calibration'])
    frozen={'normalization':contract['normalization'],'thresholds':{p.trace.A:contract['threshold']}}
    published=p.classify(raw['test'],contract['normalization'],contract['threshold'])
    args={k:d[k] for k in ('root','checkpoint','recovered_path','records','previous','identity','git','handoff')}
    args.update(config={'feature_dimension':768,'num_neighbors':9},preparation=prep,development=dev,
                reviewed={'calibration.json':frozen,'scores.json':{p.trace.A:published},'raw-scores.json':raw},
                expected_bank={'shape':[9,768],'float32_le_sha256':'synthetic'})
    return d['source_path'],args


def execute(experiment,calls,**options):
    path,args=experiment
    with zipfile.ZipFile(path) as z:
        return run.execute(source=z,**args,runtime_factory=lambda cfg:RegionFake(calls,**options))


def test_paired_run_seals_normals_before_test_and_no_mask_access(experiment,monkeypatch):
    args=experiment[1];original=run.ev.recovered_bytes
    def recovery(*a):
        assert (args['root']/'contract/calibration.json').exists()
        return original(*a)
    monkeypatch.setattr(run.ev,'recovered_bytes',recovery)
    original_read=zipfile.ZipFile.read
    def read(z,name,*a,**kw):
        assert 'ground_truth/' not in str(name)
        return original_read(z,name,*a,**kw)
    monkeypatch.setattr(zipfile.ZipFile,'read',read)
    calls=[];r=execute(experiment,calls)
    assert r['status']=='region_search_evaluated_exploratory'
    assert calls.count('features')==24 and calls.count('region')==24+9*12
    assert calls.count('predict')==25+9*12
    assert r['shared_maximum_replay_difference']==0 and r['original_photos']==4
    assert r['training_performed'] is r['existing_threshold_changed'] is r['automatic_pass_allowed'] is False
    assert r['models'][p.ARMS[0]]['fixed_threshold']['counts']==r['models'][p.ARMS[1]]['fixed_threshold']['counts']
    assert len(r['normal_gates'])==len(r['candidate_gates'])==1
    assert r['latency_ms']['test']['shared_A']['count']==4
    assert calls[-1]=='release'
    with zipfile.ZipFile(args['handoff']) as z:
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert set(z.namelist())==set(run.RESULT_FILES)
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        assert not any(n.endswith(('.npz','.pt','.png')) for n in z.namelist())
        assert '새로 놓친 불량' in z.read('index.html').decode()
        assert json.loads(z.read('candidate-calibration.json'))['test_scores_used']==0
    before=args['handoff'].read_bytes();calls=[]
    assert execute(experiment,calls)==r and calls==[] and args['handoff'].read_bytes()==before


def test_interrupted_origin_scan_reuses_only_complete_photos(experiment):
    with pytest.raises(RuntimeError,match='feature interruption'):
        execute(experiment,[],feature_fail_after=12)
    path=experiment[1]['root']/'sources/photo-001/origins.json';before=path.read_bytes()
    calls=[];r=execute(experiment,calls)
    assert calls.count('features')==12 and path.read_bytes()==before
    assert r['mixed_attempts'] and len(r['interrupted_attempts'])==1


def test_interrupted_scoring_reuses_completed_samples_and_repeats_gates(experiment):
    with pytest.raises(RuntimeError,match='scoring interruption'):
        execute(experiment,[],region_fail_after=24+12)
    path=experiment[1]['root']/'development/photo-001/sample.json';before=path.read_bytes()
    calls=[];r=execute(experiment,calls)
    assert calls.count('features')==0 and calls.count('region')==24+8*12
    assert path.read_bytes()==before and r['mixed_attempts'] and len(r['candidate_gates'])==2


@pytest.mark.parametrize('name',['sources/photo-001/origins.json','development/photo-001/sample.json',
    'test/photo-001/maps.npz','contract/calibration.json','routing/routing.json','result/comparison.json'])
def test_modified_cache_rejected_before_gpu(experiment,name):
    execute(experiment,[])
    target=experiment[1]['root']/name;target.write_bytes(b'changed');calls=[]
    with pytest.raises(ValueError,match='incomplete or changed'):
        execute(experiment,calls)
    assert calls==[] and target.read_bytes()==b'changed'


def test_candidate_gate_changed_rejected_before_gpu(experiment):
    execute(experiment,[])
    next(experiment[1]['root'].glob('attempt-*/candidate-gate.json')).write_bytes(b'changed')
    calls=[]
    with pytest.raises(ValueError,match='Missing contributing candidate'):
        execute(experiment,calls)
    assert calls==[]


def test_completed_test_receipt_bound_to_normal_contract(experiment):
    execute(experiment,[])
    path=experiment[1]['root']/'test/photo-001/receipt.json'
    receipt=json.loads(path.read_bytes());receipt.pop('receipt_sha256')
    receipt['contract_sha256']='different'
    receipt['receipt_sha256']=p.fit.digest(receipt)
    path.write_text(json.dumps(receipt))
    calls=[]
    with pytest.raises(ValueError,match='different candidate contract'):
        execute(experiment,calls)
    assert calls==[]


def test_unresolved_sources_stop_before_candidate_or_test(experiment):
    calls=[]
    with pytest.raises(ValueError,match='Unresolved bank vectors'):
        execute(experiment,calls,missing=True)
    assert 'region' not in calls and not experiment[1]['handoff'].exists()


def test_shared_gate_drift_stops_before_source_or_candidate(experiment):
    calls=[]
    with pytest.raises(ValueError,match='Shared normal gate failed'):
        execute(experiment,calls,offset=1)
    assert 'features' not in calls and 'region' not in calls


def test_changed_bank_blocks_retries(experiment):
    with pytest.raises(ValueError,match='Bank changed'):
        execute(experiment,[],mutate=True)
    assert (experiment[1]['root']/'integrity-block.json').exists()
    calls=[]
    with pytest.raises(ValueError,match='Prior bank integrity'):
        execute(experiment,calls)
    assert calls==[]


def test_routing_preserves_all_aliases_and_original_bank_order(experiment):
    prep=experiment[1]['preparation'];fake=RegionFake([]);hashes=fake.all_bank_hashes();chunks=[]
    for pid in prep['training_photo_ids']:
        origins=[]
        for row in prep['training_rows']:
            if row['photo_id']==pid:
                features,indices=fake.candidate_features(None)
                origins+=p.trace.find_origins(features,indices,row,set(hashes))
        chunks.append({'photo_id':pid,'origins':origins,'candidate_embeddings_checked':12288})
    # Repeated bank entries retain distinct row indices; all 12 source aliases remain eligible.
    route,aliases=p.make_routing(hashes+[hashes[0]],chunks,prep)
    assert route['multi_region_vectors']==10 and route['origin_records']==2*12*9
    assert all(v==list(range(10)) for v in route['region_indices'].values())
    assert len(aliases[hashes[0]])==24
    bad=copy.deepcopy(chunks);bad[0]['origins'][0]['grid_x']=1
    with pytest.raises(ValueError,match='Invalid/duplicate'):
        p.make_routing(hashes,bad,prep)
    bad=copy.deepcopy(chunks);bad[0]['origins'].append(bad[0]['origins'][0])
    with pytest.raises(ValueError,match='Invalid/duplicate'):
        p.make_routing(hashes,bad,prep)
    with pytest.raises(ValueError,match='Insufficient'):
        p.make_routing(hashes,chunks,prep,minimum=10)
    # An additional distinct feature has a source ONLY in the first region.
    extra=p.trace.vector_hashes(np.full((1,768),11,dtype=np.float32))[0]
    row=prep['training_rows'][0]
    chunks[0]['origins']+=p.trace.find_origins(np.full((1,768),11,dtype=np.float32),[20],row,{extra})
    route,_=p.make_routing(hashes+[extra],chunks,prep)
    assert route['region_indices'][p.fit.REGIONS[0]]==list(range(10))
    assert all(route['region_indices'][r]==list(range(9)) for r in p.fit.REGIONS[1:])


def test_candidate_contract_rejects_test_or_overlap_and_shared_flip(experiment):
    raw=experiment[1]['reviewed']['raw-scores.json']
    with pytest.raises(ValueError,match='separate unique normals'):
        p.calibrate(raw['development'],raw['test'])
    bad=copy.deepcopy(raw['calibration']);bad[0]['photo_id']=raw['development'][0]['photo_id']
    with pytest.raises(ValueError,match='separate unique normals'):
        p.calibrate(raw['development'],bad)
    expected=raw['calibration'][1];sample=copy.deepcopy(expected)
    sample['regions'][0]['top1']+=1
    with pytest.raises(ValueError,match='score drift'):
        p.check_shared(sample,expected,experiment[1]['reviewed']['calibration.json'])
    sample=copy.deepcopy(expected)
    for row in sample['regions']: row['top1']-=1e-6
    with pytest.raises(ValueError,match='decision flip'):
        p.check_shared(sample,expected,experiment[1]['reviewed']['calibration.json'])


class Tensor:
    def __init__(self,a):self.a=np.array(a);self.device='cpu'
    def index_select(self,axis,ids):return Tensor(self.a[ids])


def test_native_bank_filter_restores_exact_object_even_after_exception():
    original=Tensor([[0,0],[1,1],[8,8],[9,9]])
    core=SimpleNamespace(memory_bank=original)
    torch=SimpleNamespace(long='int64',tensor=lambda ids,**kw:np.array(ids))
    with pytest.raises(RuntimeError,match='inference error'):
        with selected_bank(core,[2,3],torch):
            assert np.array_equal(core.memory_bank.a,[[8,8],[9,9]])
            # Nearest search is genuinely restricted rather than relabelling the full result.
            assert np.min(np.linalg.norm(core.memory_bank.a-[0,0],axis=1))>0
            raise RuntimeError('inference error')
    assert core.memory_bank is original
    runtime=RegionRuntime.__new__(RegionRuntime)
    runtime.config={'num_neighbors':9}
    with pytest.raises(ValueError,match='too small'):
        runtime.predict_region(None,list(range(8)))


def test_cli_help_has_no_gpu_import():
    script=Path(run.__file__)
    result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--archive' in result.stdout


def test_new_sources_do_not_change_legacy_fit_evaluation_identity():
    root=Path(run.__file__).resolve().parents[1]
    frozen=run.ev.run_identity(root)['source_files_lf_sha256']
    for name in ('compare_mpi_regions.py','mpi_region_runtime.py','mpi_region_search_protocol.py'):
        assert (root/'scripts'/name).is_file() and f'scripts/{name}' not in frozen
    launcher=(root/'scripts/run_mpi_region_search.ps1').read_text()
    assert 'scripts\\compare_mpi_regions.py' in launcher and '$LASTEXITCODE -ne 0' in launcher
    assert '.venv\\Scripts\\python.exe' in launcher and 'Remove-Item' not in launcher
