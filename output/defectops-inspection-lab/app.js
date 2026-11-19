import { cases, model, threshold, stages, verdicts, sop, route, outcomes } from './cases.js';
import { readRecord, emptyRecord, saveAction, latest, stageStates, canIssue, reviewer, recordKey } from './store.js';
import { drawSpecimen, icon } from './graphics.js';

const $ = selector => document.querySelector(selector);
const pad = n => String(n).padStart(2,'0');
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const params = new URLSearchParams(location.search);
let caseId = Object.hasOwn(cases, params.get('case')) ? params.get('case') : '01';
let stage = /^[1-6]$/.test(params.get('step')) ? Number(params.get('step'))-1 : 0;
let historyTab = 'events', busy = false, scene, toastTimer, editing = false;
let record = load();
const specimen = () => cases[caseId];
const revName = review => 'R' + pad(review.number);
function timestamp(iso) {
  const d=new Date(iso);
  return d.getFullYear()+'.'+pad(d.getMonth()+1)+'.'+pad(d.getDate())+' '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());
}
function announce(text) {
  clearTimeout(toastTimer);$('#toast').textContent=text;
  toastTimer=setTimeout(()=>{$('#toast').textContent='';},4200);
}
function storageNotice(text='') {
  const el=$('#storage-error');el.textContent=text;el.hidden=!text;
}
function load() {
  try { const data=readRecord(caseId);storageNotice();return data; }
  catch(e){storageNotice(e.message);return emptyRecord(caseId);}
}
function keepLocation(){
  const url=new URL(location.href);url.searchParams.set('case',caseId);url.searchParams.set('step',stage+1);
  history.replaceState(null,'',url);
}
function go(next, focus=false) {
  stage=next;editing=false;record=load();keepLocation();render();
  if(focus)$('#panel-title').focus({preventScroll:true});
}
function historyFocus(tab) {
  historyTab=tab;renderHistory();
  $('#history').scrollIntoView({block:'start'});$('#tab-'+tab).focus({preventScroll:true});
}
function renderCases(){
  $('#cases').innerHTML=Object.values(cases).sort((a,b)=>a.id.localeCompare(b.id)).map(item=>
    `<button class="case-option" type="button" data-case="${item.id}" aria-pressed="${item.id===caseId}">
      <canvas class="case-thumb" data-thumb="${item.id}" width="138" height="108" aria-hidden="true"></canvas>
      <span class="case-text"><span class="case-num">CASE ${item.id}</span><strong>${item.title}</strong></span>
      <span class="case-check">${icon('check',13)}</span></button>`).join('');
  document.querySelectorAll('[data-thumb]').forEach(c=>drawSpecimen(c,c.dataset.thumb,{thumbnail:true}));
}
function renderSteps(){
  const states=stageStates(record,specimen());
  $('#steps').innerHTML=stages.map((name,i)=>{
    const state=states[i], done=state==='done';
    const label=done?'완료':state==='skip'?'해당 없음':'대기';
    return (i?'<span class="step-separator" aria-hidden="true"></span>':'')+
      `<button type="button" data-step="${i}" class="step" data-state="${state}" ${i===stage?'aria-current="step"':''} aria-label="${i+1}단계 ${name} · ${label}">
        <span class="step-number">${done?icon('check',14):state==='skip'?'–':pad(i+1)}</span>${name}</button>`;
  }).join('');
  scene?.update(caseId,stage,states,latest(record)?.decision?.value);
}
function picture(map=false, single=false) {
  const item=specimen();
  return `<figure class="specimen ${single?'single':''}"><canvas width="${single?620:320}" height="${single?245:236}" data-image="${map?'map':'raw'}" role="img" aria-label="${map?item.mapAlt:item.rawAlt}"></canvas>
    <figcaption class="sample-tag">${map?'합성 이상 지도':'합성 원본'}</figcaption>${single?`<span class="sample-foot">Q1 / FRONT</span>`:''}</figure>`;
}
function drawPictures(root){
  root.querySelectorAll('[data-image]').forEach(c=>{
    const figure=c.parentElement, width=figure.clientWidth, height=figure.clientHeight;
    if(!width||!height)return;
    const density=Math.min(window.devicePixelRatio||1,2);
    c.width=Math.round(width*density);c.height=Math.round(height*density);
    drawSpecimen(c,caseId,{map:c.dataset.image==='map'});
  });
}
function pair(){return `<div class="pair">${picture()}${picture(true)}</div>`;}
function compare(){
  return `<div class="compare-head"><strong>원본·지도 비교</strong><button type="button" class="small-link" data-do="enlarge">${icon('enlarge',14)}크게 보기</button></div>${pair()}`;
}
function primary(action,label,{disabled=false,lead='',arrow=true}={}){
  return `<button type="button" class="primary wide" data-do="${action}" ${disabled?'disabled':''}><span class="btn-leading">${lead?icon(lead,16):''}${label}</span>${arrow?icon('arrow',17):''}</button>`;
}
function nextButton(i,title){return primary('go-'+i,title);}
function gate(text){return `<div class="gate">${icon('lock',16)}<span>${text}</span></div>`;}
function note(text,title=''){
  return `<div class="callout">${icon('info',16)}<div>${title?`<strong>${title}</strong>`:''}${text}</div></div>`;
}
function panelHead(status,desc,tone=''){
  return `<div class="panel-meta"><span class="eyebrow">STEP ${pad(stage+1)} / 06</span><span class="status ${tone}"><i></i>${status}</span></div>
    <h2 id="panel-title" tabindex="-1">${stages[stage]}</h2><p class="panel-desc">${desc}</p>`;
}
function inputPanel(){
  const item=specimen();
  return panelHead(record.registeredAt?'검토 대상 등록됨':'등록 대기','저장된 트랜지스터 원본을 열어 검토 대상으로 등록하세요.',record.registeredAt?'good':'')+
    picture(false,true)+`<dl class="metadata"><dt>검사 ID</dt><dd class="id">${item.inspection}</dd><dt>품목</dt><dd>트랜지스터</dd><dt>시연용 모델</dt><dd class="id">${model}</dd></dl>`+
    primary('register',record.registeredAt?'등록된 검사 다시 열기':'시연 시작',{lead:record.registeredAt?'file':'play'});
}
function aiPanel(){
  const item=specimen(), warning=item.score>=threshold;
  return panelHead(record.viewedAt?'결과 확인됨':'고정 결과 재생','합성 지도에서 모델이 반응한 위치를 원본과 비교하세요.',record.viewedAt?'good':'')+
    compare()+`<div class="scorebox"><div><span>예시 이상 점수</span><strong>${item.score.toFixed(2)}</strong></div><div><span>경고 기준</span><strong class="threshold">${threshold.toFixed(2)}</strong></div><b class="score-state ${warning?'':'low'}">${warning?'모델 경고':'기준 미만'}</b></div>`+
    note(item.reading)+`<p class="small-note">점수는 확인 순서를 돕는 예시입니다. 제품 합격이나 불량을 확정하지 않습니다.</p>`+
    (!record.registeredAt?gate('이미지 입력에서 검사를 등록하면 결과를 확인 처리할 수 있습니다.'):'')+
    primary('view','결과 확인하고 검토하기',{disabled:!record.registeredAt});
}
function reviewPanel(r){
  const ready=Boolean(record.registeredAt && record.viewedAt);
  return panelHead(r?revName(r)+' 저장됨':'관찰 의견 작성','원본에서 본 형상과 사진만으로 확인하기 어려운 점을 구분해 기록하세요.',r?'good':'')+
    compare()+`<form id="review-form"><fieldset><legend>관찰 결과</legend><div class="choices">${
      Object.entries(verdicts).map(([k,v])=>`<label class="choice"><input name="verdict" type="radio" value="${k}" required ${r?.verdict===k?'checked':''}><span>${v}</span></label>`).join('')}</div></fieldset>
    <div class="form-tools"><span class="small-note">${reviewer}${r?' · '+revName(r):''}</span><button type="button" class="small-link" data-do="example">${icon('pen',14)}예시 의견 불러오기</button></div>
    <label class="field">관찰 내용<textarea name="note" required maxlength="1200" rows="2" placeholder="원본에서 보이는 위치와 형상을 적으세요.">${escape(r?.note)}</textarea></label>
    <label class="field">판단 한계 · 재확인할 점<textarea name="limit" ${r?.verdict==='hold'?'required':''} maxlength="800" rows="2" placeholder="판단을 보류한다면 이유를 적으세요.">${escape(r?.limit)}</textarea></label>
    ${ready?'':gate('검사 등록과 AI 결과 확인을 마친 뒤 의견을 저장할 수 있습니다.')}
    <button class="primary wide" type="submit" ${ready?'':'disabled'}><span>${r?'새 검토 개정 저장':'검토 의견 저장'}</span>${icon('arrow',17)}</button></form>`+
    (r?`<button type="button" class="small-link" data-do="go-${route(r,specimen())==='sop'?3:5}">저장한 개정으로 계속 ${icon('chevron',13)}</button>`:'');
}
function outcomeBox(branch){
  const [title,desc]=outcomes[branch];
  return note(desc,title)+`<button type="button" class="secondary wide" data-do="history-reviews">검토 개정 보기 ${icon('history',16)}</button>`+
    `<button type="button" class="small-link" data-do="go-2">검토 의견 수정 ${icon('chevron',13)}</button>`;
}
function sopPanel(r){
  const branch=route(r,specimen());
  let html=panelHead(r?revName(r)+' 기준':'검토 의견 대기','SOP는 작업 절차서입니다. 관찰한 리드 형상에 해당하는 모의 문서의 원문을 현재 검토 개정에 연결합니다.');
  if(outcomes[branch])return html+outcomeBox(branch);
  if(branch==='pending')return html+gate('먼저 작업자 검토에서 관찰 의견을 저장하세요.')+primary('evidence','현재 개정에 근거 연결',{disabled:true})+nextButton(2,'작업자 검토 열기');
  const doc=r.evidence??sop;
  html+=`<article class="document"><span class="eyebrow">MOCK SOP · r${doc.version}</span><h3>${doc.title}</h3><p>적용 범위 · ${doc.scope}</p><p class="ref-code">${doc.source}</p><blockquote>${doc.text}</blockquote><p>${doc.limit}</p></article>`;
  return html+(r.evidence?note(revName(r)+'에 원문과 문서 개정을 저장했습니다.','근거 연결 완료')+nextButton(4,'모의 조치와 승인으로 이동'):primary('evidence','현재 개정에 근거 연결',{lead:'book'}));
}
function approvalPanel(r){
  const branch=route(r,specimen());
  let html=panelHead(r?revName(r)+' 기준':'초안 대기','추가 확인 요청을 초안으로 남기고, 시연 승인 담당자가 승인 또는 반려를 기록합니다.');
  if(outcomes[branch])return html+outcomeBox(branch);
  if(!r?.evidence)return html+gate('현재 검토 개정에 SOP 근거를 연결하면 초안을 작성할 수 있습니다.')+primary('draft','추가 확인 요청 초안 작성',{disabled:true})+nextButton(r?3:2,r?'SOP 근거 열기':'작업자 검토 열기');
  html+=`<p class="small-note">연결 근거 <span class="ref-code">${r.evidence.source}</span></p>`;
  if(!r.draft)return html+`<form id="draft-form"><label class="field">추가 확인 요청<textarea name="text" rows="3" maxlength="800" required>왼쪽 리드의 형상을 다른 각도의 사진으로 다시 확인해 주세요.</textarea></label><button type="submit" class="primary wide"><span>초안 작성</span>${icon('pen',16)}</button></form>`+
    `<p class="small-note">초안을 저장하면 모의 승인과 반려를 선택할 수 있습니다.</p>`;
  html+=`<article class="document"><span class="eyebrow">모의 조치 초안</span><h3>${r.draft.title}</h3><p>${escape(r.draft.text)}</p><span class="small-note">${r.draft.actor} · ${timestamp(r.draft.at)}</span></article>`;
  if(!r.decision)return html+`<p class="small-note">승인 담당자 (시연) · 실제 인증·권한 검증 없음</p><div class="panel-actions"><button type="button" class="secondary" data-do="reject">모의 반려</button><button type="button" class="primary" data-do="approve">${icon('check',16)}모의 승인</button></div>`;
  const approved=r.decision.value==='approved';
  return html+note(`${r.decision.actor} · ${timestamp(r.decision.at)}`,approved?'모의 승인 완료':'모의 반려 · 티켓 생성 차단')+
    (approved?nextButton(5,'모의 티켓 열기'):nextButton(2,'검토 개정 다시 작성'));
}
function ticketPanel(r){
  const branch=route(r,specimen());
  let html=panelHead(r?revName(r)+' 기준':'승인 대기','현재 개정의 근거·초안·승인을 한 장의 모의 티켓에 연결합니다. 외부 전송 없음.');
  if(outcomes[branch])return html+outcomeBox(branch);
  if(!canIssue(r)){
    const rejected=r?.decision?.value==='rejected';
    return html+gate(rejected?'현재 초안이 반려되었습니다. 새 검토 개정에서 다시 시작하세요.':'현재 검토 개정에 연결된 근거·초안·모의 승인이 모두 필요합니다.')+
      primary('ticket','모의 티켓 생성',{disabled:true})+nextButton(rejected||!r?2:r.evidence?4:3,rejected||!r?'작업자 검토 열기':r.evidence?'모의 승인 열기':'SOP 근거 열기');
  }
  if(r.ticket)html+=`<article class="ticket-result">${icon('ticket',22)}<h3>모의 티켓 생성 완료</h3><span class="ref-code">${r.ticket.id}</span><p>${revName(r)} · ${r.evidence.source} · 모의 승인</p><p>이 브라우저에만 저장 · 외부 전송 없음</p></article>`;
  else html+=note(`${revName(r)} · ${r.evidence.title} r${r.evidence.version} · 추가 확인 요청`,'모의 승인 확인');
  return html+primary('ticket',r.ticket?'동일 티켓 다시 확인':'모의 티켓 생성',{lead:'ticket'})+
    `<div class="form-tools"><button type="button" class="small-link" data-do="go-2">검토 의견 수정</button><button type="button" class="small-link" data-do="history-reviews">연결 이력 보기 ${icon('chevron',13)}</button></div>`;
}
function renderPanel(){
  const review=latest(record), views=[inputPanel,aiPanel,reviewPanel,sopPanel,approvalPanel,ticketPanel];
  const panel=$('#panel');panel.dataset.stage=String(stage);panel.dataset.revision=String(review?.number??0);
  panel.dataset.registration=record.registeredAt??'';
  panel.innerHTML=views[stage](review);drawPictures(panel);
}
function entries(){
  const list=[];
  const add=(at,title,detail)=>list.push({at,title,detail});
  if(record.registeredAt)add(record.registeredAt,'검토 대상 등록',specimen().inspection);
  if(record.viewedAt)add(record.viewedAt,'합성 AI 결과 확인',model+' · 예시 점수 '+specimen().score.toFixed(2));
  for(const r of record.reviews){
    const rev=revName(r);
    add(r.at,'검토 의견 저장',rev+' · '+verdicts[r.verdict]);
    if(r.evidence)add(r.evidence.at,'모의 SOP 연결',rev+' · '+r.evidence.source);
    if(r.draft)add(r.draft.at,'추가 확인 요청 초안',rev+' · '+r.draft.actor);
    if(r.decision)add(r.decision.at,r.decision.value==='approved'?'모의 승인':'모의 반려',rev+' · '+r.decision.actor);
    if(r.ticket)add(r.ticket.at,'모의 티켓 생성',rev+' · '+r.ticket.id);
  }
  return list.reverse();
}
function revisionDetails(r){
  const branch=route(r,specimen());
  const chain=(name,body)=>`<div class="rev-chain"><h4>${name}</h4><p>${body}</p></div>`;
  let html=`<dl><dt>검사 ID</dt><dd class="id">${specimen().inspection}</dd><dt>모델 결과</dt><dd>${model} · 예시 ${specimen().score.toFixed(2)} / 기준 ${threshold.toFixed(2)}</dd><dt>작성자</dt><dd>${r.actor}</dd><dt>관찰 내용</dt><dd>${escape(r.note)}</dd><dt>판단 한계</dt><dd>${escape(r.limit)||'별도 기록 없음'}</dd></dl>`;
  if(outcomes[branch])html+=chain(outcomes[branch][0],outcomes[branch][1]);
  if(r.evidence){
    const d=r.evidence;
    html+=chain('연결한 모의 SOP',`${d.title} · 문서 r${d.version} · ${d.scope}<br><span class="id">${d.source}</span> · ${d.text}<br><span class="id">${d.limitSource}</span> · ${d.limit}<br>${d.actor} · ${timestamp(d.at)}`);
  }
  if(r.draft)html+=chain('모의 조치 초안',`${r.draft.title} · ${escape(r.draft.text)}<br>${r.draft.actor} · ${timestamp(r.draft.at)}`);
  if(r.decision)html+=chain('초안에 대한 결정',`${r.decision.value==='approved'?'모의 승인':'모의 반려'} · ${r.decision.actor} · ${timestamp(r.decision.at)}`);
  if(r.ticket)html+=chain('연결된 모의 티켓',`<span class="id">${r.ticket.id}</span> · ${timestamp(r.ticket.at)} · 외부 전송 없음`);
  return html;
}
function renderHistory(){
  const events=entries(),revs=record.reviews;
  $('#tab-events span').textContent=events.length;$('#tab-reviews span').textContent=revs.length;
  for(const tab of ['events','reviews'])$('#tab-'+tab).setAttribute('aria-pressed',String(tab===historyTab));
  const empty=text=>`<div class="empty-history">${icon('list',22)}<p>${text}</p></div>`;
  let html;
  if(historyTab==='events'){
    html=events.length?`<ol class="event-log">${events.map((e,i)=>`<li><span class="event-index">${pad(events.length-i)}</span><strong>${e.title}</strong><span class="detail">${escape(e.detail)}</span><time class="time" datetime="${e.at}">${timestamp(e.at)}</time></li>`).join('')}</ol>`:
      empty('아직 처리 기록이 없습니다. 시연을 시작하면 이 사례의 검사와 검토 이력이 여기에 쌓입니다.');
  }else{
    html=revs.length?revs.slice().reverse().map((r,i)=>`<details class="revision" ${i===0?'open':''} data-revision="${r.number}"><summary><span class="rev-code">${revName(r)}</span>${verdicts[r.verdict]}${i===0?'<small class="current-rev">현재 개정</small>':''}<time class="time" datetime="${r.at}">${timestamp(r.at)}</time>${icon('down',15)}</summary><div class="rev-body">${revisionDetails(r)}</div></details>`).join(''):
      empty('저장된 검토 개정이 없습니다. 작업자 검토에서 관찰 의견을 저장하세요.');
  }
  $('#history-body').innerHTML=html;
}
function render(){
  renderSteps();renderPanel();renderHistory();
  $('#mobile-start-label').textContent=record.registeredAt?'현재 작업 열기':'시연 시작';
}
$('#mobile-start').onclick=async()=>{
  if(!record.registeredAt)await commit('register');
  if(record.registeredAt){
    $('#panel').scrollIntoView({block:'start'});
    $('#panel-title').focus({preventScroll:true});
  }
};
async function commit(action,input={}){
  if(busy)return;
  const savedCase=caseId, fromStage=stage;
  const registeredBefore=Boolean(record.registeredAt), ticketBefore=Boolean(latest(record)?.ticket);
  const expected={number:Number($('#panel').dataset.revision),registration:$('#panel').dataset.registration};
  busy=true;$('#panel').setAttribute('aria-busy','true');
  try{
    const updated=await saveAction(savedCase,action,expected,input);
    if(savedCase!==caseId)return;
    record=updated;editing=false;storageNotice();
    const r=latest(record);
    const messages={
      register:registeredBefore?'이미 등록된 검사를 다시 열었습니다.':'검토 목록에 합성 검사를 등록했습니다.',view:'합성 결과를 확인했습니다.',
      review:'검토 의견을 새 개정으로 저장했습니다.',evidence:'현재 개정에 SOP 원문을 연결했습니다.',
      draft:'추가 확인 요청 초안을 저장했습니다.',approve:'현재 초안에 모의 승인을 기록했습니다.',reject:'모의 반려를 기록했습니다.',
      ticket:r?.ticket?(ticketBefore?'기존 티켓을 다시 열었습니다. ':'모의 티켓을 생성했습니다. ')+r.ticket.id:'',reset:'선택한 사례의 기록을 초기화했습니다.',
    };
    if(action==='reset'){stage=0;historyTab='events';}
    else if(stage===fromStage){
      if(action==='register')stage=1;
      if(action==='view')stage=2;
      if(action==='review')stage=route(r,specimen())==='sop'?3:5;
    }
    keepLocation();render();$('#panel-title').focus({preventScroll:true});announce(messages[action]);
  }catch(e){
    if(savedCase===caseId){storageNotice(e.message);announce(e.message);}
  }finally{busy=false;$('#panel').removeAttribute('aria-busy');}
}
function fillExample(){
  const f=$('#review-form'), example=specimen().example;
  f.elements.verdict.value=example.verdict;f.elements.note.value=example.note;f.elements.limit.value=example.limit;
  f.elements.limit.required=example.verdict==='hold';editing=true;f.elements.note.focus({preventScroll:true});
  announce('예시 의견을 불러왔습니다. 내용을 확인하고 저장하세요.');
}
$('#panel').addEventListener('input',e=>{
  editing=true;
  if(e.target.name==='verdict')$('#review-form').elements.limit.required=e.target.value==='hold';
});
$('#panel').addEventListener('submit',e=>{
  e.preventDefault();const f=e.target;const data=Object.fromEntries(new FormData(f));
  if(f.id==='review-form')commit('review',data);
  if(f.id==='draft-form')commit('draft',data);
});
$('#panel').addEventListener('click',e=>{
  const button=e.target.closest('[data-do]');if(!button||button.disabled)return;
  const action=button.dataset.do;
  if(action.startsWith('go-'))return go(Number(action.slice(3)),true);
  if(action==='example')return fillExample();
  if(action==='history-reviews')return historyFocus('reviews');
  if(action==='enlarge'){
    $('#image-content').innerHTML=pair()+note(specimen().reading);
    $('#image-dialog').showModal();drawPictures($('#image-content'));return;
  }
  commit(action);
});
$('#cases').addEventListener('click',e=>{
  const b=e.target.closest('[data-case]');if(!b||b.dataset.case===caseId||busy)return;
  caseId=b.dataset.case;historyTab='events';renderCases();go(0);$('[data-case="'+caseId+'"]').focus({preventScroll:true});
});
$('#steps').addEventListener('click',e=>{
  const b=e.target.closest('[data-step]');if(!b)return;go(Number(b.dataset.step));$('[data-step="'+stage+'"]').focus({preventScroll:true});
});
for(const t of ['events','reviews'])$('#tab-'+t).onclick=()=>{historyTab=t;renderHistory();};
$('#reset-open').onclick=()=>{
  $('#reset-description').textContent='CASE '+caseId+' · '+specimen().title+'의 등록, 모든 검토 개정과 모의 티켓을 이 브라우저에서 삭제합니다.';
  $('#reset-dialog').showModal();
};
$('#reset-confirm').onclick=()=>{$('#reset-dialog').close();commit('reset');};
window.addEventListener('storage',e=>{
  if(e.key!==null && e.key!==recordKey(caseId))return;
  if(editing){
    storageNotice('다른 탭에서 사례 기록이 바뀌었습니다. 현재 단계 버튼을 다시 누르면 최신 기록을 엽니다. 작성 중인 의견은 먼저 복사해 두세요.');
  }else{record=load();render();}
});
window.addEventListener('resize',()=>drawPictures(document));
renderCases();render();
const { startScene, usePlan } = await import('./scene.js');
const selectZone=i=>go(i);
try { scene=startScene($('#viewport'),selectZone); }
catch { scene=usePlan($('#viewport'),selectZone); }
scene.update(caseId,stage,stageStates(record,specimen()),latest(record)?.decision?.value);
