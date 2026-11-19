import { isInspectionId, readInspection, readTrialReport } from './live-api.js';

const count = value => value.toLocaleString('ko-KR');
const hash = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

function matches(photo, record, trial) {
  return photo.inspection_id === record.inspection_id && record.environment?.device === 'cpu' &&
    photo.input_sha256 === record.input?.sha256 && photo.heatmap_sha256 === record.artifacts?.heatmap_sha256 &&
    photo.cpu_score === record.raw_score && photo.cpu_area === record.pixel?.anomalous_map_pixels &&
    trial.pixel_threshold === record.pixel?.threshold && trial.model_image_threshold === record.image_threshold &&
    trial.checkpoint_sha256 === record.trace?.checkpoint_sha256 && trial.manifest_sha256 === record.trace?.manifest_sha256 &&
    trial.evaluation_id === record.trace?.evaluation_id;
}

export async function readAreaReview(id, inspection) {
  const trial = await readTrialReport(id);
  const { anchor, cases, map_size_hw: shape } = trial;
  const invalid = () => new Error('두 사진의 이상 지도와 면적 실험 기록이 일치하지 않습니다.');
  if (trial.type !== 'map_area' || trial.schema_version !== 1 || trial.trial_id !== id ||
      !Array.isArray(shape) || shape.length !== 2 || !shape.every(n => Number.isInteger(n) && n > 0 && n % 2 === 0) ||
      !Array.isArray(cases) || !cases.length ||
      !Number.isFinite(trial.score_threshold) || !Number.isFinite(trial.lower_score_threshold) ||
      !(trial.area_fraction > 0 && trial.area_fraction <= 1) || !Number.isFinite(trial.pixel_threshold)) throw invalid();
  const pixels = shape[0] * shape[1];
  const validPhoto = photo => isInspectionId(photo?.inspection_id) && /^[a-z][a-z0-9_]*\/\d+\.png$/.test(photo.sample) &&
    [photo.input_sha256, photo.area_sha256, photo.heatmap_sha256].every(hash) &&
    ['cpu','gpu'].every(d => {
      const area = photo[d + '_area'], right = photo[d + '_right'], score = photo[d + '_score'];
      return Number.isFinite(score) && score >= trial.lower_score_threshold && score < trial.score_threshold &&
        Number.isInteger(area) && area >= 0 && area <= pixels && Number.isInteger(right) && right >= 0 &&
        right <= Math.min(area, pixels / 2);
    });
  const counts = trial.counts?.score_or_area;
  if (!validPhoto(anchor) || anchor.ground_truth_label !== 1 ||
      !cases.every(p => validPhoto(p) && p.ground_truth_label === 0) ||
      new Set([anchor, ...cases].map(p => p.inspection_id)).size !== cases.length + 1 ||
      trial.area_cut !== Math.ceil(trial.area_fraction * pixels) ||
      !['cpu','gpu'].every(d => ['tp','fp','fn','tn'].every(k => Number.isInteger(counts?.[d]?.[k]) && counts[d][k] >= 0) &&
        Object.values(counts[d]).reduce((a, b) => a + b, 0) === trial.sample_count)) throw invalid();
  const selected = cases.find(p => p.inspection_id === inspection.inspection_id);
  if (!selected || !matches(selected, inspection, trial) || !matches(anchor, await readInspection(anchor.inspection_id), trial)) throw invalid();
  return { ...trial, selected };
}

export function areaComparison(trial, showHeatmap, figure) {
  const { anchor, selected, cases, area_cut, map_size_hw: [height, width] } = trial;
  const summary = trial.counts.score_or_area;
  const resultText = d => `미탐 ${summary[d].fn}장 · 오경보 ${summary[d].fp}장`;
  const deviceRates = resultText('cpu') === resultText('gpu') ? `CPU·GPU 모두 ${resultText('cpu')}` :
    `CPU ${resultText('cpu')} / GPU ${resultText('gpu')}`;
  const options = cases.map(p => `<option value="${p.inspection_id}" ${p === selected ? 'selected' : ''}>${p.sample}</option>`);
  function photoCard(photo, kind, label) {
    const area = photo.cpu_area;
    const flagged = photo.cpu_score >= trial.score_threshold || area >= area_cut;
    return `<div class="area-photo">${figure(kind, `${label} · ${photo.sample}`)}
      <p class="area-values"><strong>전체 ${count(area)}픽셀 (${(100 * area / (height * width)).toFixed(2)}%)</strong>
      <span>오른쪽 절반 ${count(photo.cpu_right)}픽셀</span>
      <span>점수 ${photo.cpu_score.toFixed(5)} · <b class="${flagged ? 'trial-alert' : ''}">${flagged ? '시험 경고' : '시험 경고 없음'}</b></span></p></div>`;
  }
  const legend = showHeatmap ? `두 이상 지도의 색상 범위는 0부터 약 ${trial.pixel_threshold.toFixed(5)}까지입니다.` :
    `분홍색은 픽셀 기준(약 ${trial.pixel_threshold.toFixed(5)}) 이상인 영역입니다. 세로 점선은 사진의 중앙입니다.`;
  return `<div class="threshold-trial area-intro"><div class="trial-intro"><div>
    <h3>점수와 면적을 함께 적용한 결과</h3><p>테스트 ${trial.sample_count}장 · ${deviceRates}</p></div>
    <label>비교할 정상 사진 · ${cases.length}장 <select id="area-case">${options.join('')}</select></label></div>
    <p class="area-rule">사진 점수가 ${trial.score_threshold} 이상이거나, 이상 영역이 지도 전체의 ${trial.area_fraction * 100}% 이상이면 경고</p></div>
    <p class="area-legend">맥북 CPU 지도 · ${legend}</p>
    <div class="live-pictures area-pictures">
      ${photoCard(anchor, showHeatmap ? 'anchorHeatmap' : 'anchorArea', '불량')}
      ${photoCard(selected, showHeatmap ? 'heatmap' : 'normalArea', '정상')}</div>`;
}
