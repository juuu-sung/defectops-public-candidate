"""No human opinions are written to real case DBs; all writes use synthetic evidence."""
import copy
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from defectops.input_review import Conflict, Evidence, Notes, create_app


@pytest.fixture
def evidence():
    regions = [dict(region=f'grid1-{i}_s1024', state='NEEDS_INPUT_REVIEW',
                    statistics={'connected_black_pixels': i}, source={'member_sha256': str(i)*64}) for i in (1,2,3)]
    e = SimpleNamespace(digest='a'*64, report_sha='b'*64, policy_sha='c'*64,
                        cases={'validation/0001': dict(state='NEEDS_INPUT_REVIEW', regions=regions,
                               model_result={'alert':True}, human_assessment='NOT_RECORDED')},
                        images={f'images/validation-0001-grid1-{i}_s1024.png': b'png' for i in (1,2)})
    e.image_name = lambda case, region: f'images/{case.replace("/", "-")}-{region}.png'
    return e


@pytest.fixture
def notes(tmp_path, evidence):
    return Notes(tmp_path/'notes.sqlite3', evidence)


def fields():
    return dict(author='테스트 작성자', observation='테스트용 관찰', scope_status='UNKNOWN',
                scope='', reference='', result='UNDETERMINED', reason='필수 부위 미정', change_reason='')


def save(notes, data=None, base=0, request_id=None):
    return notes.save('validation/0001', data or fields(), ['grid1-1_s1024'], base, request_id or uuid4().hex)


def test_append_restart_export_and_model_preserved(notes,evidence):
    before=copy.deepcopy(evidence.cases)
    first=save(notes)
    data=fields(); data.update(observation='정정 관찰',change_reason='표현 보완')
    second=save(notes,data,1)
    reloaded=Notes(notes.path,evidence)
    assert [r['revision'] for r in reloaded.history('validation/0001')]==[1,2]
    assert second['previous_revision']==1 and first['observation']=='테스트용 관찰'
    assert second['bundle_sha256']==evidence.digest and second['region_sources']
    assert second['automatic_pass_allowed'] is False and second['production_approved'] is False
    assert evidence.cases==before


def test_idempotency_conflict_and_required_revision_reason(notes):
    token=uuid4().hex
    first=save(notes,request_id=token)
    assert save(notes,request_id=token)==first
    with pytest.raises(Conflict): save(notes)
    with pytest.raises(ValueError,match='수정 이유'): save(notes,base=1)
    data=fields();data['observation']='changed'
    with pytest.raises(Conflict):save(notes,data,request_id=token)
    assert len(notes.all_records())==1


@pytest.mark.parametrize('change',[
    {'author':''},{'observation':''},{'reason':''},{'author':'x'*81},
    {'observation':'x'*2001},{'scope_status':'invalid'},{'result':'PASS'},
    {'result':'CONFIRMED'},{'scope_status':'DOCUMENTED'},
    {'scope_status':'DOCUMENTED','scope':'부품'}])
def test_invalid_fields(notes,change):
    data=fields();data.update(change)
    with pytest.raises(ValueError):save(notes,data)
    assert notes.all_records()==[]


def test_confirmed_requires_image_and_not_blank(notes,evidence):
    data=fields(); data.update(scope_status='DOCUMENTED',scope='기재한 부품',reference='테스트 문서 1절',result='CONFIRMED')
    assert save(notes,data)['result']=='CONFIRMED'
    data['change_reason']='타일 변경'
    evidence.cases['validation/0001']['regions'][0]['state']='INSUFFICIENT_INFORMATION'
    with pytest.raises(ValueError,match='전체가 검은'):save(notes,data,1)
    with pytest.raises(ValueError,match='원본이 없는'):
        notes.save('validation/0001',data,['grid1-3_s1024'],1,uuid4().hex)


@pytest.mark.parametrize('regions',[[],['bad'],['grid1-1_s1024']*2])
def test_invalid_region(notes,regions):
    with pytest.raises(ValueError): notes.save('validation/0001',fields(),regions,0,uuid4().hex)


def test_database_separation_and_sql_append_only(notes,evidence,tmp_path):
    save(notes)
    with sqlite3.connect(notes.path) as con:
        for query in ['DELETE FROM input_review_revisions', 'UPDATE input_review_revisions SET revision=3',
                      'DELETE FROM input_review_meta']:
            with pytest.raises(sqlite3.IntegrityError):con.execute(query)
    changed=copy.copy(evidence);changed.digest='d'*64
    with pytest.raises(ValueError,match='다른 입력'): Notes(notes.path,changed)
    other=tmp_path/'production.sqlite3'
    with sqlite3.connect(other) as con:con.execute('CREATE TABLE inspections (id TEXT)')
    with pytest.raises(ValueError,match='다른 용도'):Notes(other,evidence)
    with sqlite3.connect(other) as con:
        assert con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()==[('inspections',)]


def form(client):
    page=client.get('/case/validation/0001')
    hidden=dict(re.findall(r'name="(csrf|base|request_id)" value="([^"]+)"',page.text))
    return dict(fields(),**hidden,regions='grid1-1_s1024')


def test_http_form_history_and_security(notes,evidence):
    client=TestClient(create_app(evidence,notes),base_url='http://127.0.0.1:8770')
    assert client.get('/').status_code==200
    data=form(client); data['observation']='<script>alert(1)</script>'
    response=client.post('/case/validation/0001',data=data,follow_redirects=False)
    assert response.status_code==303
    assert client.post('/case/validation/0001',data=data,follow_redirects=False).status_code==303
    assert len(notes.all_records())==1
    content=client.get(response.headers['location']).text
    assert '<script>' not in content and '&lt;script&gt;' in content
    assert '의견을 저장했습니다' in content and '기존 모델 경고' in content
    assert 'name="author" maxlength="80" required value=""' in content
    export=client.get('/export').json()
    assert len(export['records'])==1 and not export['automatic_pass_allowed']
    assert client.post('/case/validation/0001',data={**data,'csrf':'bad'}).status_code==403
    assert client.post('/case/validation/0001',data={**data,'csrf':'한글'}).status_code==403
    assert client.post('/case/validation/0001',data=data,headers={'Origin':'https://evil.example'}).status_code==403
    assert client.get('/',headers={'Host':'evil.example'}).status_code==403
    assert client.get('/image/validation/0001/grid1-1_s1024').content==b'png'
    assert client.get('/image/validation/0001/bad').status_code==404
    assert client.get('/case/validation/9999').status_code==404


def test_validation_preserves_form_and_stale_write(notes,evidence):
    client=TestClient(create_app(evidence,notes),base_url='http://127.0.0.1:8770')
    data=form(client)
    bad={**data,'result':'CONFIRMED'}
    response=client.post('/case/validation/0001',data=bad)
    assert response.status_code==422 and '테스트용 관찰' in response.text
    assert not notes.all_records()
    save(notes)
    assert client.post('/case/validation/0001',data=data).status_code==409
    assert len(notes.all_records())==1


def test_wrong_archive_refused(tmp_path):
    archive=tmp_path/'bad.zip';archive.write_bytes(b'wrong archive')
    with pytest.raises(ValueError,match='고정된'):Evidence(archive)


def test_parallel_edit_only_one_wins(notes):
    from concurrent.futures import ThreadPoolExecutor
    def attempt(_):
        try: return save(notes)['revision']
        except Conflict: return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(attempt,range(2)))
    assert set(result)=={1,'conflict'} and len(notes.all_records())==1


def test_failed_write_keeps_input(notes,evidence,monkeypatch):
    client=TestClient(create_app(evidence,notes),base_url='http://127.0.0.1:8770')
    data=form(client)
    def fail(*args): raise sqlite3.OperationalError('database locked')
    monkeypatch.setattr(notes,'save',fail)
    response=client.post('/case/validation/0001',data=data)
    assert response.status_code==503 and '테스트용 관찰' in response.text
    assert not notes.all_records()


def test_archive_snapshot_verification(tmp_path,monkeypatch):
    import hashlib
    import zipfile
    from defectops import input_review as mod
    report={'mode':'archive','production_approved':False,'policy':{'automatic_pass_allowed':False},
            'photos':[{'group':'training','photo_id':str(i)} for i in range(512)]}
    payload={'input-assessment.json':json.dumps(report).encode(),'images/example.png':b'png'}
    manifest={n:hashlib.sha256(raw).hexdigest() for n,raw in payload.items()}
    archive=tmp_path/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for n,raw in payload.items():z.writestr(n,raw)
        z.writestr('handoff-manifest.json',json.dumps(manifest))
    monkeypatch.setattr(mod,'EXPECTED_ZIP_SHA',hashlib.sha256(archive.read_bytes()).hexdigest())
    e=Evidence(archive)
    assert len(e.cases)==512 and e.images['images/example.png']==b'png'
    archive.write_bytes(b'changed')
    assert e.images['images/example.png']==b'png'
    with pytest.raises(ValueError):Evidence(archive)
