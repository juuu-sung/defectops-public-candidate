import { cases, route, sop, verdicts } from './cases.js';

// Browser-only persistence boundary. A future API adapter must use server IDs and revisions.
const prefix = 'defectops.inspection-lab.v1.';
export const recordKey = id => prefix + id;
export const emptyRecord = id => ({ version: 1, caseId: id, registeredAt: null, viewedAt: null, reviews: [] });
export const latest = record => record.reviews.at(-1);
export const reviewer = '품질 담당자 (시연)';
export const approver = '승인 담당자 (시연)';

function requireState(test, message) {
  if (!test) throw new Error(message);
}

export function readRecord(id) {
  let raw;
  try { raw = localStorage.getItem(recordKey(id)); }
  catch { throw new Error('브라우저 저장소를 열 수 없습니다. 브라우저의 사이트 데이터 설정을 확인하세요.'); }
  if (raw === null) return emptyRecord(id);
  try {
    const value = JSON.parse(raw);
    if (value.version !== 1 || value.caseId !== id || !Array.isArray(value.reviews)) throw new Error();
    return value;
  } catch { throw new Error('저장 기록을 읽을 수 없습니다. 이 사례의 기록을 초기화하면 다시 시작할 수 있습니다.'); }
}

export function canIssue(review) {
  if (!review?.evidence || !review.draft || review.decision?.value !== 'approved') return false;
  return review.evidence.reviewId === review.id &&
    review.draft.evidenceId === review.evidence.id &&
    review.decision.draftId === review.draft.id;
}

function apply(record, action, expected, input) {
  const specimen = cases[record.caseId];
  const review = latest(record);
  const now = new Date().toISOString();
  if (action === 'register') {
    if (record.registeredAt) return false;
    record.registeredAt = now;
    return true;
  }
  requireState(record.registeredAt, '먼저 이미지를 검사 대상으로 등록하세요.');
  if (action === 'view') {
    if (record.viewedAt) return false;
    record.viewedAt = now;
    return true;
  }
  requireState(record.viewedAt, '먼저 AI 이상탐지에서 합성 결과를 확인하세요.');
  requireState((review?.number ?? 0) === expected.number && record.registeredAt === expected.registration,
    '검토 개정이 바뀌었습니다. 최신 기록을 확인한 뒤 다시 저장하세요.');
  if (action === 'review') {
    const note = input.note.trim();
    const limit = input.limit.trim();
    requireState(Object.hasOwn(verdicts, input.verdict) && note, '관찰 결과를 선택하고 관찰 내용을 입력하세요.');
    requireState(input.verdict !== 'hold' || limit, '판단을 보류하는 이유와 재확인할 내용을 입력하세요.');
    requireState(note.length <= 1200 && limit.length <= 800, '관찰 내용은 1,200자, 판단 한계는 800자까지 저장할 수 있습니다.');
    requireState(!review || review.verdict !== input.verdict || review.note !== note || review.limit !== limit,
      '이미 저장한 의견입니다. 내용을 수정하면 새 개정을 저장할 수 있습니다.');
    const number = expected.number + 1;
    record.reviews.push({ number, id: specimen.inspection + '-R' + number, verdict: input.verdict, note, limit, at: now, actor: reviewer });
    return true;
  }
  requireState(route(review, specimen) === 'sop', '현재 검토 개정에 적용할 모의 SOP가 없습니다.');
  if (action === 'evidence') {
    if (review.evidence) return false;
    review.evidence = { ...sop, id: review.id + '-S', reviewId: review.id, at: now, actor: reviewer };
    return true;
  }
  requireState(review.evidence?.reviewId === review.id, '현재 검토 개정에 SOP 근거를 먼저 연결하세요.');
  if (action === 'draft') {
    if (review.draft) return false;
    const text = input.text.trim();
    requireState(text && text.length <= 800, '추가로 확인할 내용을 1~800자로 입력하세요.');
    review.draft = { id: review.id + '-D', evidenceId: review.evidence.id, title: '추가 확인 요청', text, at: now, actor: reviewer };
    return true;
  }
  requireState(review.draft?.evidenceId === review.evidence.id, '현재 근거에 연결된 초안을 먼저 작성하세요.');
  if (action === 'approve' || action === 'reject') {
    requireState(!review.decision, '이미 결정이 기록된 초안입니다. 새 결정은 검토를 개정한 뒤 진행하세요.');
    review.decision = { id: review.id + '-A', draftId: review.draft.id, value: action === 'approve' ? 'approved' : 'rejected', actor: approver, at: now };
    return true;
  }
  requireState(action === 'ticket' && canIssue(review), '현재 검토 개정의 초안에 모의 승인이 필요합니다.');
  if (review.ticket) return false;
  review.ticket = { id: 'TKT-TR-' + specimen.id + '-R' + String(review.number).padStart(2, '0'), decisionId: review.decision.id, actor: reviewer, at: now };
  return true;
}

// Serialize tabs on localhost. Re-read inside the lock, and publish a result only after the write succeeds.
export async function saveAction(id, action, expected, input = {}) {
  const run = () => {
    if (action === 'reset') {
      try { localStorage.removeItem(recordKey(id)); }
      catch { throw new Error('초기화하지 못했습니다. 저장소 설정을 확인한 뒤 다시 시도하세요.'); }
      return emptyRecord(id);
    }
    const record = readRecord(id);
    if (apply(record, action, expected, input)) {
      try { localStorage.setItem(recordKey(id), JSON.stringify(record)); }
      catch { throw new Error('저장하지 못했습니다. 브라우저 저장 공간과 사이트 데이터 설정을 확인한 뒤 다시 시도하세요.'); }
    }
    return record;
  };
  return navigator.locks ? navigator.locks.request(recordKey(id), run) : run();
}

export function stageStates(record, specimen) {
  const review = latest(record);
  const branch = route(review, specimen);
  const skipped = ['closed', 'recheck', 'unmatched'].includes(branch);
  return [
    record.registeredAt, record.viewedAt, review, review?.evidence, review?.decision, review?.ticket,
  ].map((value, index) => value ? 'done' : skipped && index >= 3 ? 'skip' : 'waiting');
}
