import { icon } from './graphics.js';
import { startScene, usePlan } from './scene.js';
import { checkHealth, inspectHeader, isInspectionId, openImage, readAreaPicture, readGroundTruth, readInspection, readPicture, readThresholdTrial } from './live-api.js';
import { areaComparison, readAreaReview } from './area-review.js';

const $ = selector => document.querySelector(selector);
const escape = value => String(value ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const stages = ['이미지 입력', 'PatchCore 검사', '원본·지도 비교'];
const labels = {
  input: '원본', heatmap: '이상 지도', overlay: '원본 위 이상 지도',
  devices: '연구실 GPU와 맥북 CPU의 이상 지도 및 픽셀별 차이',
  maskInput: '원본 사진 + 정답 마스크', maskHeatmap: '맥북 이상 지도 + 정답 마스크',
  anchorArea: '불량 사진의 이상 영역', normalArea: '정상 사진의 이상 영역', anchorHeatmap: '불량 사진의 이상 지도',
};
const mapButtons = { heatmap: '이상 지도', overlay: '원본에 겹쳐 보기', devices: 'GPU · CPU 비교', truth: '정답 마스크 겹쳐 보기',
  areaMask: '이상 영역', areaHeatmap: '이상 지도' };
const registrationLabels = { REGISTERED: '검토 목록에 등록됨', PENDING: '검토 목록 등록 대기', BLOCKED: '검토 목록 등록 차단' };
const panel = $('#panel'), comparison = $('#live-comparison');
let stage = 0, scene, health, checking = false, working = false, decoding = false;
let file, preview, result, loadingImages = false, requestError = '', inputError = '';
let healthError = '', imageErrors = {}, images = {}, view = 'heatmap', generation = 0;
let monitor = { input: null, map: null }, truthReview, thresholdTrial, areaReview;
const params = new URLSearchParams(location.search);
const initialId = params.get('inspection');
let savedId = isInspectionId(initialId) ? initialId : null;
const reference = params.get('reference');
const referenceMode = reference === 'ground-truth' ? 'truth' : reference === 'gpu' ? 'devices' : null;
const referenceId = referenceMode ? savedId : null;
const trialId = savedId && !referenceMode ? params.get('trial') : null;
const areaId = savedId && !referenceMode && !trialId ? params.get('area') : null;
const comparingArea = () => Boolean(areaId && result?.inspection_id === initialId);
if (referenceId) view = referenceMode;
if (trialId) view = 'overlay';
if (areaId) view = params.get('map') === 'heatmap' ? 'areaHeatmap' : 'areaMask';

document.querySelector('.simulation').innerHTML = '<i></i>실제 사진 검사 · PatchCore';
$('.hero h1').textContent = '트랜지스터 사진 검사';
$('.hero p').textContent = '사진을 선택해 모델의 이상 점수를 확인하고, 원본과 서버가 생성한 지도를 비교하세요.';
for (const selector of ['#cases', '#history', '#mobile-start', '.duration']) $(selector).hidden = true;

function setId(id) {
  savedId = id;
  const url = new URL(location.href);
  if (id) url.searchParams.set('inspection', id);
  else {
    url.searchParams.delete('inspection');
    url.searchParams.delete('reference');
    url.searchParams.delete('trial');
    url.searchParams.delete('area');
    url.searchParams.delete('map');
    if (url.hash === '#live-comparison') url.hash = '';
    view = 'heatmap';
  }
  history.replaceState(null, '', url);
}
function resetImages() {
  for (const picture of [preview, ...Object.values(images)]) {
    if (picture) URL.revokeObjectURL(picture.url);
  }
  preview = null;
  images = {};
  imageErrors = {};
  truthReview = null;
  thresholdTrial = null;
  areaReview = null;
  monitor = { input: null, map: null };
}
function updateMonitor() {
  if (comparingArea()) {
    const heatmap = view === 'areaHeatmap';
    monitor = { input: images[heatmap ? 'anchorHeatmap' : 'anchorArea']?.image,
      map: images[heatmap ? 'heatmap' : 'normalArea']?.image,
      captions: areaReview ? [areaReview.anchor.sample, areaReview.selected.sample] : ['DEFECT', 'NORMAL'] };
    return;
  }
  const masked = view === 'truth';
  const input = result ? images[masked ? 'maskInput' : 'input'] : preview;
  const map = masked ? images.maskHeatmap : view === 'overlay' ? images.overlay : images.heatmap;
  monitor = { input: input?.image, map: map?.image, groundTruth: masked && !!images.maskHeatmap };
}
function button(action, label, secondary = false, disabled = false) {
  return `<button type="button" data-live-action="${action}" class="${secondary ? 'secondary' : 'primary'} wide" ${disabled ? 'disabled' : ''}>${label}</button>`;
}
function message(text) {
  return text ? `<p class="live-error" role="alert">${escape(text)}</p>` : '';
}
function metadata(pairs) {
  return `<dl class="metadata">${pairs.map(([k,v]) => `<dt>${k}</dt><dd>${escape(v)}</dd>`).join('')}</dl>`;
}
function status() {
  const device = health?.device;
  const deviceName = device === 'cpu' ? ' · CPU' : device === 'cuda' ? ' · CUDA GPU' : '';
  const title = checking ? 'API 연결 확인 중' : health ? '모델 준비됨' + deviceName : '검사 API 미연결';
  return `<div class="live-status"><strong class="status ${health ? 'good' : 'warn'}"><i></i>${title}</strong>
    <button class="small-link" data-live-action="health" ${checking || working ? 'disabled' : ''}>연결 다시 확인</button></div>
    ${healthError ? `<p class="small-note">${escape(healthError)} 모델 서버를 시작한 후 연결을 다시 확인하세요.</p>` : ''}`;
}
function uploadPanel() {
  const previewFigure = preview ? `<figure class="upload-preview"><img src="${preview.url}" alt="선택한 트랜지스터 사진">
    <figcaption>${escape(file.name)}<br><small>1024 × 1024 · RGB PNG</small></figcaption></figure>` : '';
  return status() + `<label class="live-file">트랜지스터 사진 선택<input id="photo" type="file" accept="image/png,.png" ${working ? 'disabled' : ''}></label>
    <p class="small-note">1024 × 1024 · RGB PNG · 최대 50 MiB</p>${previewFigure}${message(inputError)}
    ${decoding ? '<p role="status">사진을 확인하고 있습니다.</p>' : ''}
    ${result ? button('result', '현재 검사 결과 보기', true) :
      button('inspect', working ? '검사 중…' : '실제 추론 실행', false, !health || !file || working || decoding)}
    <p class="small-note">실행하면 사진과 검사 결과가 연결된 검사 서버에 저장됩니다.</p>`;
}
function scorePanel() {
  if (!result) {
    if (working) return `<div class="callout" role="status">${icon('clock')}<span>${savedId ? '검사 결과를 불러오는 중입니다.' : '서버가 사진에서 이상 점수와 지도를 계산하고 있습니다.'}</span></div>`;
    return `${savedId ? button('restore', '기존 검사 다시 조회', true) : ''}
      ${button('input', file ? '선택한 사진 열기' : '사진 선택으로 이동', true)}`;
  }
  const r = result;
  const verdict = r.decision === 'REVIEW_REQUIRED' ? '모델 경고' : '기준 미만';
  const display = value => value.toLocaleString('ko-KR', { maximumFractionDigits: 5, useGrouping: false });
  const registration = registrationLabels[r.review_registration?.status] ??
    (r.review_registration ? r.review_registration.status : '자동 등록 꺼짐');
  return `<div class="scorebox"><div><span>실제 이상 점수</span><strong title="${r.raw_score}">${display(r.raw_score)}</strong></div>
    <div><span>경고 기준</span><strong class="threshold" title="${r.image_threshold}">${display(r.image_threshold)}</strong></div>
    <b class="score-state ${r.decision === 'PASS' ? 'low' : ''}">${verdict}</b></div>
    <p class="small-note">점수는 결함 확률이 아닙니다. 경고 기준 미만인 사진도 외관 검토가 필요합니다.</p>
    ${metadata([['검사 ID',r.inspection_id],['모델',r.trace?.fit_run_id ?? r.model],['검토 등록', registration]])}
    ${button('compare', '원본·지도 비교하기')}${button('input', '다른 사진 선택', true)}`;
}
function comparePanel() {
  if (!result) return scorePanel();
  const description = comparingArea() ? '왼쪽에 놓친 불량, 오른쪽에 선택한 정상 사진의 지도를 표시합니다.' : '원본과 같은 검사 ID의 지도를 표시합니다.';
  return `<p class="callout">${loadingImages ? '서버의 검증된 PNG 산출물을 불러오고 있습니다.' : description}</p>
    ${button('show-images', '비교 화면 열기')}${button('input', '다른 사진 선택', true)}`;
}
function drawPanel() {
  panel.dataset.stage = String(stage);
  const desc = ['검사할 트랜지스터의 원본 사진을 선택하세요.',
    '서버가 이 사진에 적용한 점수와 보정 기준을 확인하세요.',
    '사진에서 보이는 형상과 모델 반응의 위치를 비교하세요.'];
  panel.innerHTML = `<span class="eyebrow">STEP 0${stage + 1} / 03</span>
    <h2 id="panel-title" tabindex="-1">${stages[stage]}</h2><p class="panel-desc">${desc[stage]}</p>
    ${[uploadPanel, scorePanel, comparePanel][stage]()}${message(requestError)}`;
}
function drawSteps() {
  const ready = comparingArea() ? images.anchorArea && images.normalArea && images.anchorHeatmap && images.heatmap : images.input && images.heatmap;
  const states = [file || result ? 'done' : 'waiting', result ? 'done' : 'waiting', ready ? 'done' : 'waiting'];
  $('#steps').innerHTML = stages.map((name,i) =>
    (i ? '<span class="step-separator" aria-hidden="true"></span>' : '') +
    `<button type="button" class="step" data-live-stage="${i}" data-state="${states[i]}" ${i === stage ? 'aria-current="step"' : ''}>
      <span class="step-number">${states[i] === 'done' ? icon('check', 14) : '0' + (i+1)}</span>${name}</button>`
  ).join('');
  scene?.update(monitor, stage, states);
}
function figure(kind, caption = labels[kind]) {
  const picture = images[kind];
  const name = escape(caption);
  const body = picture
    ? `<a href="${picture.url}" target="_blank" rel="noopener" aria-label="${name} 원본 크기로 열기"><img src="${picture.url}" alt="${caption === labels[kind] ? '검사 사진의 ' + name : name + ' · ' + labels[kind]}"></a>`
    : `<p>${escape(imageErrors[kind] || (result ? '불러오는 중…' : '검사 후 표시됩니다.'))}</p>`;
  return `<figure class="live-figure"><figcaption>${name}</figcaption>${body}</figure>`;
}
function missedCases() {
  if (!truthReview) return '';
  const links = truthReview.misses.map(({ inspection_id: id, sample }) =>
    `<a href="?mode=live&amp;inspection=${id}&amp;reference=ground-truth#live-comparison" ${id === result.inspection_id ? 'aria-current="page"' : ''}>${escape(sample)}</a>`);
  return `<nav class="truth-cases" aria-label="미탐 사진 선택"><strong>미탐 사진</strong>${links.join('')}</nav>`;
}
function maskedPictures() {
  if (!truthReview) {
    return imageErrors.truth ? message(imageErrors.truth) : '<p role="status">공식 정답 마스크를 확인하고 있습니다.</p>';
  }
  const { cpu_score, image_threshold, pixel_threshold, ground_truth_pixels, overlap_pixels } = truthReview;
  const percent = (100 * overlap_pixels / ground_truth_pixels).toFixed(1);
  return `<div class="truth-key"><span><i aria-hidden="true"></i>민트색 영역과 테두리 · 공식 정답 마스크</span>
    <span>사진 점수 ${cpu_score.toFixed(5)} / 경고 기준 ${image_threshold.toFixed(5)}</span></div>
    <div class="live-pictures truth-pictures">${figure('maskInput')}${figure('maskHeatmap')}</div>
    <p class="small-note">256 × 256 이상 지도에서 정답 결함 ${ground_truth_pixels.toLocaleString('ko-KR')}픽셀 중 ${overlap_pixels.toLocaleString('ko-KR')}픽셀(${percent}%)이 픽셀 경고 기준 ${pixel_threshold.toFixed(5)} 이상입니다.</p>`;
}
function thresholdPanel() {
  if (!trialId || result.inspection_id !== initialId) return '';
  if (!thresholdTrial) {
    return imageErrors.trial ? message(imageErrors.trial) : '<p role="status">사진 경고 기준 실험을 불러오고 있습니다.</p>';
  }
  const { selected, cases, baseline, trial, sample_count, baseline_threshold, trial_threshold } = thresholdTrial;
  const options = cases.map(item => `<option value="${item.inspection_id}" ${item === selected ? 'selected' : ''}>${escape(item.sample)}</option>`);
  const score = value => `<span title="${value}">${value.toFixed(5)}</span>`;
  return `<section class="threshold-trial" aria-labelledby="trial-title">
    <div class="trial-intro"><div><h3 id="trial-title">사진 경고 기준 실험</h3>
      <p>테스트 ${sample_count}장 · 미탐 ${baseline.cpu.fn} → ${trial.cpu.fn}장 · 오경보 ${baseline.cpu.fp} → ${trial.cpu.fp}장</p></div>
      <label>새로 경고된 정상 사진 · ${cases.length}장
        <select id="trial-case">${options.join('')}</select></label></div>
    <dl class="trial-scores">
      <div><dt>맥북 CPU 점수</dt><dd>${score(selected.cpu_score)}</dd></div>
      <div><dt>연구실 GPU 점수</dt><dd>${score(selected.gpu_score)}</dd></div>
      <div><dt>기존 기준 ${score(baseline_threshold)}</dt><dd>경고 없음</dd></div>
      <div><dt>시험 기준 ${score(trial_threshold)}</dt><dd class="trial-alert">경고 · 정답은 정상</dd></div>
    </dl></section>`;
}
function drawComparison() {
  comparison.hidden = false;
  if (!result) {
    comparison.innerHTML = `<h2 id="live-compare-title">원본·지도 비교</h2><p class="small-note">실제 추론이 완료되면 이곳에 원본과 이상 지도가 표시됩니다.</p>`;
    return;
  }
  const area = comparingArea();
  const modes = area ? ['areaMask', 'areaHeatmap'] : ['heatmap', 'overlay'];
  if (result.inspection_id === referenceId) modes.push(referenceMode);
  let pictures;
  if (area) pictures = areaReview ? areaComparison(areaReview, view === 'areaHeatmap', figure) :
    (imageErrors.area ? message(imageErrors.area) : '<p role="status">불량과 정상 사진의 비교 자료를 확인하고 있습니다.</p>');
  else if (view === 'truth') pictures = maskedPictures();
  else if (view === 'devices') pictures = '<div class="device-figures">' + figure('devices') + '</div>';
  else pictures = '<div class="live-pictures">' + figure('input') + figure(view) + '</div>';
  comparison.innerHTML = `<div class="compare-head"><h2 id="live-compare-title">${area ? '불량·정상 이상 지도 비교' : '원본·지도 비교'}</h2>
    <div class="map-switch" role="group" aria-label="지도 표시 방식">${modes.map(k =>
      `<button type="button" data-map="${k}" aria-pressed="${k === view}">${mapButtons[k]}</button>`).join('')}</div></div>
    ${missedCases()}${thresholdPanel()}${pictures}
    ${Object.keys(imageErrors).length ? button('retry-images', '실패한 자료 다시 불러오기', true, loadingImages) : ''}
    <p class="small-note">${area ? '' : '검사 ID <span class="id">' + escape(result.inspection_id) + '</span> · '}사진을 누르면 원본 크기로 열립니다.</p>`;
}
function render() { drawSteps(); drawPanel(); drawComparison(); }
function go(index, scroll = false) {
  stage = index;
  drawSteps(); drawPanel();
  if (scroll) {
    (index === 2 && result ? comparison : panel).scrollIntoView({ block: 'start' });
  }
}
async function connect() {
  if (checking) return;
  checking = true; healthError = ''; drawPanel();
  try { health = await checkHealth(); }
  catch (error) { health = null; healthError = error.message; }
  finally { checking = false; drawPanel(); }
}
async function selectFile(candidate) {
  if (working || !candidate) return;
  const token = ++generation;
  resetImages(); setId(null);
  file = null; result = null; requestError = ''; inputError = ''; decoding = true; loadingImages = false;
  render();
  try {
    await inspectHeader(candidate);
    const picture = await openImage(candidate);
    if (token !== generation) { URL.revokeObjectURL(picture.url); return; }
    preview = picture; file = candidate; updateMonitor();
  } catch (error) { if (token === generation) inputError = error.message; }
  finally { if (token === generation) { decoding = false; render(); } }
}
async function loadPictures() {
  if (!result || loadingImages) return;
  loadingImages = true; imageErrors = {};
  const token = generation, id = result.inspection_id;
  render();
  const kinds = comparingArea() ? ['heatmap'] : ['input','heatmap','overlay'];
  if (id === referenceId && referenceMode === 'devices') kinds.push('devices');
  if (id === referenceId && referenceMode === 'truth') {
    try {
      const info = truthReview ?? await readGroundTruth(result);
      if (token !== generation) return;
      truthReview = info;
      kinds.push('maskInput', 'maskHeatmap');
    } catch (error) {
      if (token !== generation) return;
      imageErrors.truth = error.message;
    }
  }
  const jobs = Object.fromEntries(kinds.map(kind => [kind, () => readPicture(id, kind, truthReview?.overlay_sha256)]));
  if (comparingArea()) {
    try {
      const info = areaReview ?? await readAreaReview(areaId, result);
      if (token !== generation) return;
      areaReview = info;
      jobs.heatmap = () => readPicture(id, 'heatmap', {'heatmap.png': info.selected.heatmap_sha256});
      jobs.anchorHeatmap = () => readPicture(info.anchor.inspection_id, 'heatmap', {'heatmap.png': info.anchor.heatmap_sha256});
      jobs.anchorArea = () => readAreaPicture(areaId, info.anchor);
      jobs.normalArea = () => readAreaPicture(areaId, info.selected);
    } catch (error) {
      if (token !== generation) return;
      imageErrors.area = error.message;
      delete jobs.heatmap;
    }
  }
  const downloads = Object.entries(jobs).filter(([kind]) => !images[kind]).map(async ([kind, load]) => {
    try {
      const picture = await load();
      if (token !== generation) { URL.revokeObjectURL(picture.url); return; }
      images[kind] = picture;
    } catch (error) {
      if (token === generation) imageErrors[kind] = `${labels[kind]}를 표시할 수 없습니다. ${error.message}`;
    }
  });
  if (trialId && id === initialId && !thresholdTrial) {
    downloads.push(readThresholdTrial(trialId, result).then(data => {
      if (token === generation) thresholdTrial = data;
    }).catch(error => {
      if (token === generation) imageErrors.trial = error.message;
    }));
  }
  await Promise.all(downloads);
  if (token !== generation) return;
  loadingImages = false;
  updateMonitor(); render();
}
async function run(restore = false) {
  if (working || (restore ? !savedId : !health || !file || result)) return;
  working = true; requestError = ''; stage = 1; render();
  try {
    result = await readInspection(restore ? savedId : file);
    setId(result.inspection_id);
  } catch (error) {
    requestError = error.message + (restore ? '' : ' 요청은 자동 재전송하지 않습니다.');
    if (!restore) stage = 0;
  } finally { working = false; render(); }
  if (result) {
    await loadPictures();
    if (restore && result?.inspection_id === initialId && stage === 1 && location.hash === '#live-comparison') go(2, true);
  }
}
function handleAction(event) {
  const b = event.target.closest('[data-live-action]');
  if (!b || b.disabled) return;
  switch (b.dataset.liveAction) {
    case 'health': return connect();
    case 'inspect': return run();
    case 'restore': return run(true);
    case 'retry-images': return loadPictures();
    case 'input': return go(0,true);
    case 'result': return go(1,true);
    case 'compare': return go(2,true);
    case 'show-images': return comparison.scrollIntoView({ block: 'start' });
  }
}
panel.addEventListener('change', e => { if (e.target.id === 'photo') selectFile(e.target.files[0]); });
panel.addEventListener('click', handleAction);
comparison.addEventListener('change', event => {
  const group = event.target.id === 'area-case' ? areaReview : event.target.id === 'trial-case' ? thresholdTrial : null;
  const id = event.target.value;
  if (!group?.cases.some(item => item.inspection_id === id)) return;
  const url = new URL(location.href);
  url.searchParams.set('inspection', id);
  url.hash = 'live-comparison';
  location.assign(url);
});
comparison.addEventListener('click', e => {
  if (e.target.closest('[data-map]')) {
    view = e.target.closest('[data-map]').dataset.map;
    if (comparingArea()) {
      const url = new URL(location.href);
      if (view === 'areaHeatmap') url.searchParams.set('map', 'heatmap');
      else url.searchParams.delete('map');
      history.replaceState(null, '', url);
    }
    updateMonitor(); drawSteps(); drawComparison();
    comparison.querySelector('[data-map="'+view+'"]').focus({preventScroll:true});
  } else handleAction(e);
});
$('#steps').addEventListener('click', e => {
  const b = e.target.closest('[data-live-stage]');
  if (b) { go(Number(b.dataset.liveStage)); $('#steps [aria-current="step"]').focus({preventScroll:true}); }
});
try { scene = startScene($('#viewport'), i=>go(i), stages); }
catch { scene = usePlan($('#viewport'), i=>go(i), undefined, stages); }
render();
connect();
if (savedId) run(true);
