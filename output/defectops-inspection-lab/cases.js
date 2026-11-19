// Fixed exhibition fixtures. This module supplies no production images or model responses.
export const model = 'PatchCore · DEMO-1';
export const threshold = 0.65;
export const stages = ['이미지 입력', 'AI 이상탐지', '작업자 검토', 'SOP 근거 연결', '모의 승인', '티켓 · 이력'];
export const verdicts = {
  observed: '이상 징후가 보임',
  clear: '눈에 띄는 이상 없음',
  hold: '판단 보류',
};

export const cases = {
  '01': {
    id: '01', inspection: 'SIM-TR-001', title: '리드 외관 이상', score: 0.82,
    rawAlt: '세 리드 중 왼쪽 리드가 바깥으로 꺾여 보이는 트랜지스터 합성 원본',
    mapAlt: '왼쪽 리드와 떨어진 몸체 상단에 주황색 반응이 집중된 합성 이상 지도',
    reading: '지도는 몸체 상단에 강하게 반응합니다. 원본에서 왼쪽 리드의 윤곽도 살펴보세요.',
    example: {
      verdict: 'observed', note: '왼쪽 리드가 휘어 보입니다. 몸체 상단의 모델 반응은 관찰한 이상 위치와 다릅니다.',
      limit: '전기적 접속 상태는 사진으로 확정할 수 없습니다.',
    },
  },
  '02': {
    id: '02', inspection: 'SIM-TR-002', title: '정상 외관 재확인', score: 0.71,
    rawAlt: '세 리드가 곧고 몸체 표면에 옅은 반사가 보이는 트랜지스터 합성 원본',
    mapAlt: '몸체 오른쪽의 표면 반사 부근에 반응한 합성 이상 지도',
    reading: '모델은 경고했지만 원본에서는 뚜렷한 외관 이상이 보이지 않습니다. 반사와 형상을 함께 살펴보세요.',
    example: {
      verdict: 'clear', note: '몸체 오른쪽에 반사가 보이지만 리드와 몸체 윤곽에서 눈에 띄는 이상은 보이지 않습니다.',
      limit: '이 사진으로는 내부 상태와 전기적 기능을 확인할 수 없습니다.',
    },
  },
  '03': {
    id: '03', inspection: 'SIM-TR-003', title: '검사 범위 가림', score: 0.43,
    rawAlt: '사선 무늬 가림판이 왼쪽 리드와 가운데 리드의 일부를 덮은 합성 원본',
    mapAlt: '가림판 경계에 약한 반응만 보이고 리드 일부는 여전히 가려진 합성 이상 지도',
    reading: '점수는 기준 미만이지만 리드 일부가 가려져 있습니다. 가려진 형상은 새 사진에서 확인해야 합니다.',
    example: {
      verdict: 'hold', note: '가림판 때문에 왼쪽과 가운데 리드의 아래쪽 형상을 볼 수 없습니다.',
      limit: '현재 사진으로는 굽힘 여부를 판단하기 어렵습니다. 리드 전체가 보이는 사진으로 재확인이 필요합니다.',
    },
  },
};

// Snapshot of the matching entry in sops/mock-catalog-v1.json.
export const sop = {
  document: 'DEMO-TR-BENT', version: 2, title: '리드 굽힘 관찰 기록',
  scope: '트랜지스터 · 리드 굽힘 의심 외관',
  source: 'DEMO-TR-BENT@r2#OBSERVE', limitSource: 'DEMO-TR-BENT@r2#LIMIT',
  text: '원본 사진에서 굽어 보이는 리드의 위치와 검사 ID를 기록한다. 오버레이의 색만으로 결함이나 전기적 접속 상태를 확정하지 않는다.',
  limit: '이 문서는 화면 흐름을 시험하기 위한 모의 근거다. 실제 생산품의 합격·폐기·설비 조치를 지시하지 않으며, 현장 조치는 별도로 승인된 문서와 담당자 확인이 필요하다.',
};

export function route(review, specimen) {
  if (!review) return 'pending';
  if (review.verdict === 'clear') return 'closed';
  if (review.verdict === 'hold') return 'recheck';
  return specimen.id === '01' ? 'sop' : 'unmatched';
}

export const outcomes = {
  closed: ['검토 종료', '외관 관찰을 마쳤습니다. 제품 합격 여부는 별도 절차에서 확인합니다.'],
  recheck: ['사진 재확인 필요', '리드 전체가 보이는 사진이 필요합니다. 판단 보류 의견을 이력에 보존합니다.'],
  unmatched: ['적용할 모의 SOP 없음', '이 사례에는 리드 굽힘 SOP를 적용할 근거가 없습니다. 관찰 내용을 보존하고 재확인을 요청하세요.'],
};
