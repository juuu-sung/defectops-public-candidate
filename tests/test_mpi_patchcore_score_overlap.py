import copy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts import review_mpi_patchcore_score_overlap as r


def fixture():
    method = {'field':'top1','scale_kind':'q90','threshold':3.,
              'regions':[{'region':x,'center':10.,'scale':2.} for x in r.fit.REGIONS]}
    def sample(key,label,score):
        return {'key':key,'photo_id':key.split('/')[1],'label':label,
                'regions':[{'region':x,'top1':score,'native_raw':score} for x in r.fit.REGIONS]}
    defect = sample('defect/0024',1,18.)
    normals = [sample(f'validation/{i:04d}',0,19.+i) for i in range(14)]
    raw = {'test':[defect],'validation':normals}
    def pred(s):
        return {'key':s['key'],'photo_id':s['photo_id'],'label':s['label'],'alert':True,
                'score':(s['regions'][0]['top1']-10)/2,'winning_region':r.fit.REGIONS[0]}
    current = {'contract.json':method,'comparison.json':{'predictions':{'candidate':{g:[pred(s) for s in ss] for g,ss in raw.items()}}}}
    return current,{},raw


def test_fixed_selection_and_all_regions_no_input_changes():
    current,parent,raw = fixture(); before=copy.deepcopy((current,raw))
    report=r.analyze(current,parent,raw)
    assert len(report['selection'])==15 and report['selection'][0]=='defect/0024'
    assert sum(len(c['tiles']) for c in report['cases'])==180
    assert report['raw_winner_at_least_defect_winner']==14
    assert (current,raw)==before and not report['contract_changed']


@pytest.mark.parametrize('change',['missing','wrong_defect'])
def test_selection_drift_rejected(change):
    current,parent,raw=fixture()
    if change=='missing':current['comparison.json']['predictions']['candidate']['validation'].pop()
    else:current['comparison.json']['predictions']['candidate']['test'][0]['key']='defect/9999'
    with pytest.raises(ValueError):r.analyze(current,parent,raw)


def test_symmetric_attribution_conserves_gap():
    a={'raw':43.,'center':33.,'scale':2.}
    b={'raw':41.,'center':34.,'scale':3.}
    out=r.decomposition(a,b); reverse=r.decomposition(b,a)
    assert sum(out['contributions'].values())==pytest.approx(5.-7./3.)
    for k,v in out['contributions'].items():assert v==pytest.approx(-reverse['contributions'][k])
    assert r.decomposition(a,a)['normal_minus_defect']==0
    for name,delta in [('raw',1.),('center',-1.),('scale',2.)]:
        changed={**a,name:a[name]+delta}
        result=r.decomposition(changed,a)
        assert result['contributions'][name]==pytest.approx(result['normal_minus_defect'])
        assert all(abs(v)<1e-12 for k,v in result['contributions'].items() if k!=name)


def png(value,mode='RGB'):
    b=io.BytesIO(); Image.new(mode,(256,256),value).save(b,format='PNG');return b.getvalue()


def make_export_fixture(tmp_path):
    current,_,raw=fixture(); report=r.analyze(current,{},raw)
    plan={'cohorts':{'test':[],'validation':[]}}
    payload={}
    for group,samples in raw.items():
        for sample in samples:
            photo={k:sample[k] for k in ('key','label','photo_id')};photo['regions']=[]
            for j,reg in enumerate(r.fit.REGIONS):
                member=f'{group}/{sample["photo_id"]}/{reg}.png'
                source=png((20,30,j));payload[member]=source
                row={'region':reg,'member':member,'member_sha256':hashlib.sha256(source).hexdigest(),
                     'decoded_rgb_sha256':hashlib.sha256(Image.open(io.BytesIO(source)).tobytes()).hexdigest()}
                if sample['label']:
                    mask_name=member+'.mask.png';mask_raw=png(255 if j==3 else 0,'L');payload[mask_name]=mask_raw
                    row.update(mask_member=mask_name,mask_sha256=hashlib.sha256(mask_raw).hexdigest(),
                               mask=r.engine.ev.mask_details(mask_raw))
                photo['regions'].append(row)
                sample['regions'][j]['member']=member
            plan['cohorts'][group].append(photo)
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
    plan['archive_sha256']=r.fit.sha256_file(archive)
    sig=r.fit.digest(plan);receipts={};checkpoint='c'*64
    for group,photos in plan['cohorts'].items():
        for i,photo in enumerate(photos):
            name=f'photo-{i:03d}'; folder=tmp_path/'artifacts/analyses'/r.p.NAME/group/name
            folder.mkdir(parents=True)
            r.fit.write_json(folder/'sample.json',raw[group][i])
            values=np.full((12,256,256),raw[group][i]['regions'][0]['top1'],dtype=np.float32)
            np.save(folder/'maps.npy',values,allow_pickle=False)
            r.fit.seal_phase(folder,name,sig,{'photo_sha256':r.fit.digest(photo),'checkpoint_sha256':checkpoint},['sample.json','maps.npy'])
            receipts[f'{group}\\{name}\\receipt.json']=json.loads((folder/'receipt.json').read_bytes())
    parent={'plan.json':plan,'fit-report.json':{'checkpoint_sha256':checkpoint},
            'evaluation/prediction-receipts.json':receipts,'evaluation/scores.json':raw}
    (tmp_path/'scripts').mkdir()
    for f in ('review_mpi_patchcore_score_overlap.py','run_mpi_patchcore_score_overlap.ps1'):
        (tmp_path/'scripts'/f).touch()
    return current,parent,report,archive


def test_export_all180_tiles_mask_semantics_repeat_and_tamper(tmp_path):
    current,parent,report,archive=make_export_fixture(tmp_path)
    output=r.export(tmp_path,archive,current,parent,report)
    before=output.read_bytes()
    with zipfile.ZipFile(output) as z:
        metadata=json.loads(z.read('review.json'))
        assert metadata['tiles']==180 and len(metadata['records'])==180
        assert len([n for n in z.namelist() if n.endswith('/mask.npy')])==12
        assert metadata['records'][3]['map_stats']['location']['gt_pixels']==65536
        assert metadata['records'][0]['map_stats']['location']['gt_pixels']==0
        for n,h in json.loads(z.read('handoff-manifest.json')).items():
            assert hashlib.sha256(z.read(n)).hexdigest()==h
    assert r.export(tmp_path,archive,current,parent,report)==output
    assert output.read_bytes()==before
    cache=tmp_path/'artifacts/analyses'/r.p.NAME/'validation/photo-000/maps.npy'
    cache.write_bytes(b'corrupt')
    with pytest.raises(ValueError):r.export(tmp_path,archive,current,parent,report)
    assert output.read_bytes()==before


def test_cli_requires_archive_unless_scores_only(monkeypatch,capsys):
    monkeypatch.setattr(r,'load',lambda _:fixture())
    assert r.main(['--scores-only'])==0
    assert 'fixed_score_overlap_diagnostic' in capsys.readouterr().out
    with pytest.raises(SystemExit):r.main([])
