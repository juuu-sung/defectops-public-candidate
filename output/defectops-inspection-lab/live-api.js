const api = './api/';
export const maxBytes = 50 * 1024 * 1024;
export const isInspectionId = id => typeof id === 'string' && /^[a-f0-9]{32}$/.test(id);
export const pictureFiles = { input: 'input.png', heatmap: 'heatmap.png', overlay: 'overlay.png' };

async function request(path, options = {}, timeout = 180_000) {
  let response;
  try {
    response = await fetch(api + path, {
      ...options, cache: 'no-store', signal: AbortSignal.timeout(timeout),
    });
  } catch {
    throw new Error('검사 서버에서 응답을 받지 못했습니다.');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = [body.detail, body.error].find(value => typeof value === 'string');
    const id = isInspectionId(body.inspection_id) ? ` · 검사 ID ${body.inspection_id}` : '';
    throw new Error((detail || `검사 요청 실패 (HTTP ${response.status})`) + id);
  }
  return response;
}

export async function checkHealth() {
  const response = await request('healthz', {}, 6500);
  const health = await response.json();
  if (health.status !== 'ready' || health.model !== 'patchcore') {
    throw new Error('연결된 서버에 PatchCore 모델이 준비되지 않았습니다.');
  }
  return health;
}

export async function readInspection(fileOrId) {
  const loading = typeof fileOrId === 'string';
  if (loading && !isInspectionId(fileOrId)) throw new Error('검사 ID 형식을 확인하세요.');
  const response = loading
    ? await request('v1/inspections/' + fileOrId)
    : await request('v1/inspections', { method: 'POST', headers: { 'Content-Type': 'image/png' }, body: fileOrId });
  const r = await response.json();
  if (r.status !== 'completed' || r.model !== 'patchcore' || r.category !== 'transistor' ||
      !isInspectionId(r.inspection_id) || (loading && r.inspection_id !== fileOrId) ||
      !Number.isFinite(r.raw_score) || !Number.isFinite(r.image_threshold) ||
      r.automatic_pass_allowed !== false ||
      r.decision !== (r.raw_score >= r.image_threshold ? 'REVIEW_REQUIRED' : 'PASS')) {
    throw new Error('완료된 트랜지스터 PatchCore 검사 응답을 확인할 수 없습니다.');
  }
  return r;
}

export async function inspectHeader(file) {
  if (!file.size || file.size > maxBytes) throw new Error('비어 있지 않은 50 MiB 이하의 PNG를 선택하세요.');
  const bytes = new Uint8Array(await file.slice(0, 33).arrayBuffer());
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length !== 33 || !signature.every((v, i) => bytes[i] === v) ||
      String.fromCharCode(...bytes.slice(12, 16)) !== 'IHDR') {
    throw new Error('이 파일은 PNG 원본이 아닙니다. RGB PNG 사진을 선택하세요.');
  }
  const head = new DataView(bytes.buffer);
  if (head.getUint32(16) !== 1024 || head.getUint32(20) !== 1024 || bytes[25] !== 2) {
    throw new Error('검사 입력은 1024 × 1024 RGB PNG입니다. 크기, 회색조 또는 알파 채널을 확인하세요.');
  }
}

export async function openImage(blob) {
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.src = url;
  try {
    await image.decode();
    return { url, image };
  } catch {
    URL.revokeObjectURL(url);
    throw new Error('PNG 이미지를 읽을 수 없습니다.');
  }
}

const comparisonPictures = {
  devices: 'gpu-comparison.png',
  maskInput: 'ground-truth/input-with-mask.png',
  maskHeatmap: 'ground-truth/heatmap-with-mask.png',
};

export async function readGroundTruth(inspection) {
  const response = await request(`v1/inspections/${inspection.inspection_id}/ground-truth/case.json`);
  const data = await response.json();
  const validHash = hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash);
  const matches = data.schema_version === 1 && data.inspection_id === inspection.inspection_id &&
    inspection.decision === 'PASS' && data.outcome === 'FN' && data.ground_truth_label === 1 &&
    data.input_sha256 === inspection.input?.sha256 && data.checkpoint_sha256 === inspection.trace?.checkpoint_sha256 &&
    data.evaluation_id === inspection.trace?.evaluation_id && data.heatmap_sha256 === inspection.artifacts?.heatmap_sha256 &&
    data.cpu_score === inspection.raw_score && data.image_threshold === inspection.image_threshold &&
    data.pixel_threshold === inspection.pixel?.threshold && validHash(data.mask_sha256) &&
    ['input-with-mask.png', 'heatmap-with-mask.png'].every(name => validHash(data.overlay_sha256?.[name])) &&
    Number.isInteger(data.ground_truth_pixels) && data.ground_truth_pixels > 0 &&
    Number.isInteger(data.overlap_pixels) && data.overlap_pixels >= 0 && data.overlap_pixels <= data.ground_truth_pixels &&
    Array.isArray(data.misses) && data.misses.every(item => isInspectionId(item?.inspection_id) && typeof item.sample === 'string') &&
    data.misses.some(item => item.inspection_id === data.inspection_id && item.sample === data.sample);
  if (!matches) throw new Error('검사 결과와 정답 마스크의 원본·모델 정보를 확인할 수 없습니다.');
  return data;
}

export async function readTrialReport(id) {
  if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(id)) throw new Error('경고 기준 실험 주소를 확인하세요.');
  const response = await request('threshold-trials/' + id);
  return response.json();
}

export async function readThresholdTrial(id, inspection) {
  const data = await readTrialReport(id);
  const { baseline_threshold: original, trial_threshold: trial, cases } = data;
  const validCase = item => isInspectionId(item?.inspection_id) && typeof item.sample === 'string' &&
    item.ground_truth_label === 0 && [item.cpu_score, item.gpu_score].every(score =>
      Number.isFinite(score) && score >= trial && score < original);
  const selected = Array.isArray(cases) && cases.find(item => item?.inspection_id === inspection.inspection_id);
  const countsMatch = phase => {
    const counts = data[phase]?.cpu;
    return ['tp', 'fn', 'fp', 'tn'].every(key => Number.isInteger(counts?.[key]) && counts[key] >= 0) &&
      counts.tp + counts.fn + counts.fp + counts.tn === data.sample_count;
  };
  if (data.schema_version !== 1 || data.trial_id !== id || data.decision_rule !== 'score >= threshold' ||
      !Number.isFinite(trial) || trial >= original || original !== inspection.image_threshold ||
      inspection.decision !== 'PASS' || inspection.environment?.device !== 'cpu' ||
      data.evaluation_id !== inspection.trace?.evaluation_id || data.checkpoint_sha256 !== inspection.trace?.checkpoint_sha256 ||
      data.manifest_sha256 !== inspection.trace?.manifest_sha256 || data.pixel_threshold !== inspection.pixel?.threshold ||
      !selected || !cases.every(validCase) || new Set(cases.map(item => item.inspection_id)).size !== cases.length ||
      selected.input_sha256 !== inspection.input?.sha256 || selected.cpu_score !== inspection.raw_score ||
      !countsMatch('baseline') || !countsMatch('trial') || data.trial.cpu.fp - data.baseline.cpu.fp !== cases.length) {
    throw new Error('사진, 모델 또는 점수가 저장된 경고 기준 실험과 일치하지 않습니다.');
  }
  return { ...data, selected };
}

export async function readPicture(id, kind, overlayHashes = {}) {
  const filename = comparisonPictures[kind] ?? 'artifacts/' + pictureFiles[kind];
  return readPng(`v1/inspections/${id}/${filename}`, overlayHashes[filename.split('/').pop()]);
}

export function readAreaPicture(trialId, photo) {
  return readPng(`threshold-trials/${trialId}/areas/${photo.inspection_id}.png`, photo.area_sha256);
}

async function readPng(path, expectedHash) {
  const response = await request(path, {}, 30_000);
  if (response.headers.get('content-type')?.split(';')[0] !== 'image/png') {
    throw new Error('서버가 PNG 이미지를 반환하지 않았습니다.');
  }
  const blob = await response.blob();
  if (expectedHash) {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer()));
    const actualHash = Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
    if (actualHash !== expectedHash) throw new Error('저장된 이미지와 해당 검사 기록이 일치하지 않습니다.');
  }
  return openImage(blob);
}
