#!/usr/bin/env python3
"""Local-only Streamlit screen for visually reviewing registered inspections."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from defectops.review_store import ReviewStore
from defectops.review_ui_data import load_review_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "artifacts" / "review" / "reviews.sqlite3"
DB_PATH = Path(os.environ.get("DEFECTOPS_REVIEW_DB", str(DEFAULT_DB)))
VERDICT_LABELS = {
    "ANOMALY_VISIBLE": "이상 징후가 보임",
    "NO_ANOMALY_VISIBLE": "눈에 띄는 이상 없음",
    "UNCERTAIN": "판단 보류",
}
MODEL_DECISION_LABELS = {
    "REVIEW_REQUIRED": "검토 필요",
    "PASS": "모델 기준 이하",
}


def _inspection_label(row: dict) -> str:
    model = row["model"] or "검사 실패"
    decision = row["model_decision"] or row["status"]
    return f"{row['inspection_id']} · {model} · {decision} · 검토 {row['review_revision']}"


def render() -> None:
    st.set_page_config(page_title="DefectOps · 작업자 검토", layout="wide")
    st.title("DefectOps 작업자 검토")
    st.caption("맥북 로컬 데모 · 모델 판정은 자동 합격이나 안전 인증이 아닙니다.")

    store = ReviewStore(DB_PATH)
    try:
        rows = store.list_inspections()
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error("검토 저장소를 열 수 없습니다. 먼저 로컬 DB를 초기화하고 검사 산출물을 등록하세요.")
        st.code(str(exc))
        return
    if not rows:
        st.info("등록된 검사가 없습니다. 검증된 검사 산출물 폴더를 먼저 등록하세요.")
        return

    labels = {row["inspection_id"]: _inspection_label(row) for row in rows}
    inspection_id = st.selectbox(
        "검사 선택",
        options=list(labels),
        format_func=lambda value: labels[value],
        key="selected_inspection",
    )
    try:
        case = load_review_case(store, inspection_id)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        st.error("산출물 무결성 검증에 실패했습니다. 이미지 표시와 검토 저장을 중단합니다.")
        st.code(str(exc))
        return

    item = case["item"]
    record = item["record"]
    st.caption(f"검사 ID: {inspection_id} · 검토 revision: {item['review_revision']}")
    saved_message = st.session_state.pop(f"saved_message_{inspection_id}", None)
    if saved_message:
        st.success(saved_message)
    if record["status"] != "completed":
        st.error("검사 실패 기록입니다. 사람의 시각 판정을 입력할 수 없으며 자동 합격도 금지됩니다.")
        st.text(f"오류 유형: {record.get('error_type', '알 수 없음')}")
        return

    st.subheader("모델 결과")
    metric_columns = st.columns([1.4, 1, 1, 0.8])
    metric_columns[0].metric("모델 표시", MODEL_DECISION_LABELS[record["decision"]])
    metric_columns[1].metric("원시 이상 점수", f"{record['raw_score']:.3f}")
    metric_columns[2].metric("고정 임곗값", f"{record['image_threshold']:.3f}")
    metric_columns[3].metric("검토 revision", item["review_revision"])
    comparison = (
        "탐색적 비교(PaDiM 평가 결과를 본 뒤 설정)"
        if record.get("comparison_status") == "exploratory_after_padim_test_observed"
        else "기준선"
    )
    st.caption(
        f"모델 원본 판정: {record['decision']} · 모델: {record['model']} · 비교 상태: {comparison}. "
        "점수는 결함 확률이나 심각도가 아닙니다."
    )

    st.subheader("이미지 확인")
    original_column, overlay_column = st.columns(2)
    with original_column:
        st.image(io.BytesIO(case["images"]["input"]), caption="검사 원본", width="stretch")
    with overlay_column:
        st.image(io.BytesIO(case["images"]["overlay"]), caption="고정 anomaly map 오버레이와 영역", width="stretch")
    with st.expander("히트맵·영역 좌표·실행 근거"):
        st.image(io.BytesIO(case["images"]["heatmap"]), caption="고정 범위 히트맵", width="stretch")
        st.json(
            {
                "pixel_regions": record.get("pixel", {}).get("regions", []),
                "trace": record.get("trace", {}),
                "input_sha256": record.get("input", {}).get("sha256"),
            }
        )

    st.subheader("사람의 검토 의견")
    if item["latest_review"]:
        latest = item["latest_review"]
        st.info(
            f"현재 기록: {VERDICT_LABELS[latest['human_verdict']]} "
            f"· revision {latest['revision']} · 검토자 {latest['reviewer']}"
        )
    else:
        st.info("아직 사람이 기록한 검토 의견이 없습니다.")
    st.caption("이 기록은 화면 관찰 의견입니다. 모델 결과를 덮어쓰거나 제품을 자동 합격 처리하지 않습니다.")

    with st.form(f"review_form_{inspection_id}"):
        reviewer = st.text_input("검토자 표기", max_chars=80, key=f"reviewer_{inspection_id}")
        verdict = st.selectbox(
            "화면에서 확인한 내용",
            options=list(VERDICT_LABELS),
            index=None,
            placeholder="판정을 선택하세요",
            format_func=lambda value: VERDICT_LABELS.get(value, "판정을 선택하세요"),
            key=f"verdict_{inspection_id}",
        )
        note = st.text_area("관찰 근거", max_chars=2000, key=f"note_{inspection_id}")
        acknowledged = st.checkbox(
            "이 의견이 자동 합격·결함 원인 확정이 아님을 확인했습니다.",
            key=f"acknowledged_{inspection_id}",
        )
        submitted = st.form_submit_button("검토 의견 저장", type="primary")

    if submitted:
        if not acknowledged or verdict is None:
            st.error("판정을 선택하고 확인 항목에 동의해야 저장할 수 있습니다.")
        else:
            try:
                event = store.add_review(
                    inspection_id,
                    expected_revision=item["review_revision"],
                    reviewer=reviewer,
                    human_verdict=verdict,
                    note=note,
                )
                st.session_state[f"saved_message_{inspection_id}"] = (
                    f"검토 의견을 revision {event['revision']}로 기록했습니다."
                )
                st.rerun()
            except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
                st.error(f"검토 저장 실패: {exc}")

    st.subheader("검토 이력")
    for event in store.review_history(inspection_id):
        st.text(
            f"#{event['revision']} · {VERDICT_LABELS[event['human_verdict']]} "
            f"· {event['reviewer']} · {event['recorded_at']}\n{event['note']}"
        )


render()
