"""API → real SQLite → real review UI, using synthetic bundles and no GPU."""
import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from defectops.inspection_api import create_app
from defectops.padim_inspection import InputValidationError
from defectops.review_registration import ReviewRegistration
from defectops.review_store import ReviewStore
from scripts.prepare_synthetic_demo import prepare


@pytest.fixture
def setup(tmp_path):
    prepare(tmp_path/'template')
    template = tmp_path/'template/synthetic-transistor-001'
    fit, evaluation, root = [tmp_path/name for name in ('fit', 'evaluation', 'inspections')]
    fit.mkdir(); evaluation.mkdir(); root.mkdir()
    db = tmp_path/'review/reviews.sqlite3'
    state = SimpleNamespace(calls=0, loads=0, root=root, db=db)
    runtime = SimpleNamespace(fit_config={'model':'patchcore'}, fit_run={'run_id':'synthetic'},
                              device=SimpleNamespace(type='cuda'),
                              setup_ms=0, torch=SimpleNamespace(cuda=SimpleNamespace(
                                  memory_allocated=lambda:0, memory_reserved=lambda:0)))

    def loader(*args):
        state.loads += 1
        return runtime

    def runner(args, **kwargs):
        state.calls += 1
        payload = args.input.read_bytes()
        if payload in (b'bad', b'crash'):
            args.output.mkdir()
            record = dict(schema_version=1, inspection_id=args.output.name,
                          status='INSPECTION_FAILED', error_type='synthetic-failure',
                          automatic_pass=False, automatic_pass_allowed=False)
        else:
            shutil.copytree(template, args.output)
            record = json.loads((args.output/'inspection.json').read_text())
            record['inspection_id'] = args.output.name
            if payload == b'below':
                record.update(raw_score=0.1, decision='PASS')
        (args.output/'inspection.json').write_text(json.dumps(record))
        if payload == b'bad':
            raise InputValidationError('synthetic invalid image')
        if payload == b'crash':
            raise RuntimeError('synthetic runtime failure')
        return record

    state.app = lambda: create_app(fit, evaluation, output_root=root, review_database=db,
                                   runtime_loader=loader, inspection_runner=runner)
    return state


def post(client, payload=b'alert', **headers):
    return client.post('/v1/inspections', content=payload,
                       headers={'content-type':'image/png', **headers})


@pytest.mark.parametrize('payload,code', [(b'alert',201),(b'below',201),(b'bad',422),(b'crash',500)])
def test_result_and_failures_automatically_reach_store(setup, payload, code):
    with TestClient(setup.app()) as client:
        response = post(client,payload)
        assert response.status_code == code
        data=response.json(); inspection_id=data['inspection_id']
        assert data['review_registration']['status']=='REGISTERED'
        saved=ReviewStore(setup.db).get_inspection(inspection_id)
        assert saved['review_revision']==0
        assert saved['record']['automatic_pass_allowed'] is False
        if payload==b'below': assert saved['model_decision']=='PASS'
        original=(setup.root/inspection_id/'inspection.json').read_bytes()
        assert 'review_registration' not in json.loads(original)
        for _ in range(2):
            assert client.post(f'/v1/inspections/{inspection_id}/review-registration').status_code==200
        assert setup.calls==1
        assert len(ReviewStore(setup.db).list_inspections())==1
        assert (setup.root/inspection_id/'inspection.json').read_bytes()==original


def test_api_result_is_visible_and_reviewable_in_actual_ui(setup):
    with TestClient(setup.app()) as client:
        inspection_id=post(client).json()['inspection_id']
    script=Path(__file__).resolve().parents[1]/'scripts/review_ui.py'
    with patch.dict('os.environ',{'DEFECTOPS_REVIEW_DB':str(setup.db)}):
        app=AppTest.from_file(script,default_timeout=10).run()
        assert not app.exception
        assert app.selectbox[0].value==inspection_id
        app.text_input[0].set_value('synthetic-api-reviewer')
        app.selectbox[1].set_value('UNCERTAIN')
        app.text_area[0].set_value('Synthetic integration check')
        app.checkbox[0].set_value(True)
        app.button[0].click().run()
        assert not app.exception
    assert len(ReviewStore(setup.db).review_history(inspection_id))==1


def test_store_failure_preserves_inspection_and_retry_does_not_infer(setup):
    with TestClient(setup.app()) as client:
        with patch.object(ReviewStore,'register',side_effect=sqlite3.OperationalError('locked')):
            data=post(client).json()
            assert data['status']=='completed'
            assert data['review_registration']['status']=='PENDING'
            inspection_id=data['inspection_id']
            retry=client.post(f'/v1/inspections/{inspection_id}/review-registration')
            assert retry.status_code==503
        stored=client.get(f'/v1/inspections/{inspection_id}').json()
        assert stored['review_registration']['status']=='PENDING'
        assert client.post(f'/v1/inspections/{inspection_id}/review-registration').status_code==200
        assert client.get(f'/v1/inspections/{inspection_id}').json()['review_registration']['status']=='REGISTERED'
        assert setup.calls==1


def test_restart_recovers_saved_unregistered_result_and_preserves_reviews(setup):
    with TestClient(setup.app()) as client:
        with patch.object(ReviewStore,'register',side_effect=sqlite3.OperationalError('locked')):
            inspection_id=post(client).json()['inspection_id']
    with TestClient(setup.app()) as client:
        assert client.get('/healthz').json()['review_registration']['startup_recovery']['REGISTERED']==1
        assert client.get(f'/v1/inspections/{inspection_id}').json()['review_registration']['status']=='REGISTERED'
        ReviewStore(setup.db).add_review(inspection_id,expected_revision=0,reviewer='demo',
                                        human_verdict='UNCERTAIN',note='Keep this revision')
    with TestClient(setup.app()): pass
    assert setup.calls==1
    assert len(ReviewStore(setup.db).review_history(inspection_id))==1


def test_tampering_and_id_conflict_are_blocked_without_overwriting(setup):
    with TestClient(setup.app()) as client:
        inspection_id=post(client).json()['inspection_id']
        record=setup.root/inspection_id/'inspection.json'
        saved=ReviewStore(setup.db).get_inspection(inspection_id)['inspection_json_sha256']
        value=json.loads(record.read_text());value['raw_score']=0.9
        record.write_text(json.dumps(value))
        assert client.post(f'/v1/inspections/{inspection_id}/review-registration').status_code==409
        assert ReviewStore(setup.db).get_inspection(inspection_id)['inspection_json_sha256']==saved
        (setup.root/inspection_id/'overlay.png').write_bytes(b'tampered')
        assert client.get(f'/v1/inspections/{inspection_id}').json()['review_registration']['status']=='BLOCKED'
        assert setup.calls==1


def test_boundary_failures_registered_without_running_model(setup):
    with TestClient(setup.app()) as client:
        with patch('defectops.inspection_api.MAX_UPLOAD_BYTES',2):
            for response in [post(client,b'bad'),post(client,b'x',**{'content-type':'text/plain'})]:
                assert response.status_code in (413,415)
                assert response.json()['review_registration']['status']=='REGISTERED'
        assert setup.calls==0
        assert len(ReviewStore(setup.db).list_inspections())==2
        assert client.post('/v1/inspections/not-uuid/review-registration').status_code==404


def test_partial_and_linked_bundles_not_silently_registered(setup,tmp_path):
    incomplete=setup.root/uuid4().hex;incomplete.mkdir()
    (incomplete/'inspection.json').write_text('{')
    external=tmp_path/'outside';external.mkdir()
    linked=setup.root/uuid4().hex;linked.symlink_to(external,target_is_directory=True)
    registration=ReviewRegistration(setup.root,setup.db)
    registration.initialize()
    assert registration.reconcile()=={'REGISTERED':0,'PENDING':0,'BLOCKED':2}
    assert not registration.store.list_inspections()


def test_wrong_database_fails_before_loading_gpu(setup):
    setup.db.parent.mkdir()
    with sqlite3.connect(setup.db) as c: c.execute('CREATE TABLE unrelated (value TEXT)')
    with pytest.raises(ValueError,match='another application'):
        with TestClient(setup.app()): pass
    assert setup.loads==0
    with sqlite3.connect(setup.db) as c:
        assert c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()==[('unrelated',)]


def test_offline_registration_cli_never_reruns_inspection(setup,capsys):
    from scripts.register_api_inspection import main
    with TestClient(setup.app()) as client:
        with patch.object(ReviewStore,'register',side_effect=sqlite3.OperationalError('locked')):
            inspection_id=post(client).json()['inspection_id']
    args=['--inspection-id',inspection_id,'--output-root',str(setup.root),'--review-db',str(setup.db)]
    assert main(args)==0
    assert main(args)==0
    assert 'REGISTERED' in capsys.readouterr().out
    assert setup.calls==1
    assert len(ReviewStore(setup.db).list_inspections())==1


def test_server_launcher_enables_registration_unless_explicitly_disabled():
    from scripts.serve_patchcore import main
    # The model itself is never initialized by this CLI wiring check.
    with patch('scripts.serve_patchcore.create_app') as create, patch('scripts.serve_patchcore.uvicorn.run'):
        main(['--fit-run','fit','--evaluation','evaluation'])
        assert create.call_args.kwargs['review_database'].name=='reviews.sqlite3'
        main(['--fit-run','fit','--evaluation','evaluation','--no-review-registration'])
        assert create.call_args.kwargs['review_database'] is None
