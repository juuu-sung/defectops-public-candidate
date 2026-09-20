#!/usr/bin/env python3
"""Local-only Streamlit screen for visually reviewing registered inspections."""

from __future__ import annotations

import io
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from defectops.mock_sop import DEFECT_LABELS, find_mock_sop, load_mock_catalog
from defectops.review_store import ReviewStore
from defectops.review_ui_data import load_review_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "artifacts" / "review" / "reviews.sqlite3"
MOCK_CATALOG = PROJECT_ROOT / "sops" / "mock-catalog-v1.json"
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
SEOUL = ZoneInfo("Asia/Seoul")


def _display_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(SEOUL).strftime("%Y-%m-%d %H:%M:%S KST")


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
        if DB_PATH.is_file():
            store.initialize()
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
            f"· {event['reviewer']} · {_display_time(event['recorded_at'])}\n{event['note']}"
        )

    st.subheader("모의 SOP 근거 연결")
    st.caption(
        "화면 흐름을 시험하기 위한 가상 문서입니다. 기존 관찰 문장에서 결함 유형이나 원인을 "
        "자동 추측하지 않으며, 실제 현장 절차로 사용할 수 없습니다. 검색 결과를 확인한 뒤 "
        "별도로 저장해야 검토 revision에 연결됩니다."
    )
    latest = item["latest_review"]
    if latest is None:
        st.info("사람의 검토 의견이 저장된 뒤에만 모의 문서를 찾을 수 있습니다.")
        return
    if latest["human_verdict"] != "ANOMALY_VISIBLE":
        st.info("최신 검토 의견이 '이상 징후가 보임'이 아니므로 모의 조치 근거를 제시하지 않습니다.")
        return

    evidence_history = store.sop_evidence_history(inspection_id)
    current_evidence = next(
        (event for event in evidence_history if event["review_revision"] == item["review_revision"]),
        None,
    )
    if current_evidence is not None:
        st.success(
            f"연결 기록 완료 · 검토 revision {current_evidence['review_revision']} · "
            f"{DEFECT_LABELS[current_evidence['defect_type']]} · "
            f"{current_evidence['document_id']} 개정 {current_evidence['document_revision']}"
        )
        st.caption(
            f"선택 기록자: {current_evidence['selected_by']} · 검색 기준일: "
            f"{current_evidence['lookup_date']} · 기록 시각: "
            f"{_display_time(current_evidence['recorded_at'])}"
        )
        for source in current_evidence["sources"]:
            st.info(source["text"])
            st.caption(f"저장된 원문 근거: {source['citation_id']}")
        st.warning("이 기록은 실제 제품 판정·설비 조치·티켓 발행 승인이 아닙니다.")
        return

    with st.form(f"mock_sop_form_{inspection_id}_{item['review_revision']}"):
        defect_type = st.selectbox(
            "사람이 직접 선택한 의심 유형",
            options=list(DEFECT_LABELS),
            format_func=lambda value: DEFECT_LABELS[value],
            key=f"mock_sop_type_{inspection_id}_{item['review_revision']}",
        )
        requested = st.form_submit_button("모의 SOP 근거 찾기")
    result_key = f"mock_sop_result_{inspection_id}_{item['review_revision']}"
    if requested:
        try:
            catalog = load_mock_catalog(MOCK_CATALOG)
            result = find_mock_sop(
                catalog,
                product=record["category"],
                defect_type=defect_type,
                human_verdict=latest["human_verdict"],
                as_of=date.today(),
            )
            st.session_state[result_key] = result
        except (OSError, ValueError) as exc:
            st.session_state.pop(result_key, None)
            st.error(f"유효한 모의 문서를 확인할 수 없어 검색 결과를 표시하지 않습니다: {exc}")
            return
    result = st.session_state.get(result_key)
    if result is None:
        return
    if result["status"] == "defect_type_required":
        st.warning("유형을 모르면 문서를 임의로 연결하지 않습니다. '모름'을 유지하고 확인을 보류하세요.")
        return
    if result["status"] != "found_simulation_only":
        st.warning("해당 품목·유형·날짜에 유효한 모의 문서가 없습니다. 조치를 제안하지 않습니다.")
        return

    st.success(f"모의 문서: {result['title']} · 개정 {result['revision']}")
    st.caption(
        f"검색 기준: {result['product']} · {DEFECT_LABELS[result['defect_type']]} "
        f"· 기준일 {result['as_of']} · 유효 기간 {result['valid_from']}~"
        f"{result['valid_to'] or '종료일 없음'}"
    )
    for source in result["sources"]:
        st.info(source["text"])
        st.caption(f"원문 근거: {source['citation_id']}")
    st.warning("이 내용으로 실제 제품 판정·설비 조치·티켓 발행을 하지 않습니다.")
    with st.expander("모의 문서 파일 무결성 표시"):
        st.code(f"SHA-256: {result['catalog_sha256']}")
    with st.form(f"mock_sop_save_form_{inspection_id}_{item['review_revision']}"):
        selected_by = st.text_input(
            "유형 선택 기록자",
            value=latest["reviewer"],
            max_chars=80,
            key=f"mock_sop_selected_by_{inspection_id}_{item['review_revision']}",
        )
        evidence_acknowledged = st.checkbox(
            "이 연결은 모의 근거 기록이며 실제 원인 확정이나 조치 승인이 아님을 확인했습니다.",
            key=f"mock_sop_ack_{inspection_id}_{item['review_revision']}",
        )
        save_evidence = st.form_submit_button("이 모의 근거 연결 기록 저장", type="primary")
    if save_evidence:
        if not evidence_acknowledged:
            st.error("확인 항목에 동의해야 연결 기록을 저장할 수 있습니다.")
            return
        try:
            event = store.add_sop_evidence(
                inspection_id,
                expected_review_revision=item["review_revision"],
                selected_by=selected_by,
                defect_type=result["defect_type"],
                result=result,
            )
            st.session_state.pop(result_key, None)
            st.session_state[f"saved_message_{inspection_id}"] = (
                f"모의 SOP 근거를 검토 revision {event['review_revision']}에 연결했습니다."
            )
            st.rerun()
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            st.error(f"모의 SOP 근거 연결 저장 실패: {exc}")


render()
