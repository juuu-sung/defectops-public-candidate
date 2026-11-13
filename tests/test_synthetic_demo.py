"""The public demo must stay synthetic and never overwrite a prior review."""
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from scripts.prepare_synthetic_demo import prepare
from defectops.review_store import ReviewStore


def test_demo_is_verified_explicitly_synthetic_and_non_destructive(tmp_path):
    output = tmp_path/'demo'
    db = prepare(output)
    store = ReviewStore(db)
    item = store.get_inspection('synthetic-transistor-001')
    assert item['record']['source_kind'] == 'SYNTHETIC_DEMO'
    assert item['record']['model_inference_performed'] is False
    assert store.verify(item['inspection_id'])['status'] == 'verified'
    store.add_review(item['inspection_id'], expected_revision=0, reviewer='demo-reviewer',
                     human_verdict='UNCERTAIN', note='Synthetic observation only')
    with pytest.raises(FileExistsError):
        prepare(output)
    assert len(store.review_history(item['inspection_id'])) == 1


def test_demo_banner_and_real_review_revision_flow(tmp_path):
    db = prepare(tmp_path/'demo')
    script = Path(__file__).resolve().parents[1]/'scripts/review_ui.py'
    with patch.dict('os.environ', {'DEFECTOPS_REVIEW_DB': str(db)}):
        app = AppTest.from_file(script, default_timeout=10).run()
        assert not app.exception
        assert any('합성 데모' in w.value for w in app.warning)
        assert app.metric[1].label == '모의 점수'
        app.text_input[0].set_value('demo-reviewer')
        app.selectbox[1].set_value('UNCERTAIN')
        app.text_area[0].set_value('합성 그림의 왼쪽 다리 위치를 확인합니다.')
        app.checkbox[0].set_value(True)
        app.button[0].click().run()
        app.text_area[0].set_value('합성 그림의 위치 차이만 기록하며 접속 상태는 판단하지 않습니다.')
        app.button[0].click().run()
        assert not app.exception
        assert app.metric[3].value == '2'
    history = ReviewStore(db).review_history('synthetic-transistor-001')
    assert [e['revision'] for e in history] == [1, 2]
    assert history[0]['note'] != history[1]['note']
