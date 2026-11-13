"""Loopback-only research input notes, separate from model and production reviews."""
from __future__ import annotations

import hashlib
import hmac
import html
import io
import json
from pathlib import Path
import secrets
import sqlite3
from datetime import datetime, timezone
from urllib.parse import parse_qs
from uuid import uuid4
import zipfile

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

EXPECTED_ZIP_SHA = '00f6c69801af9c26a078028e89bcecfc3392dceb0c7b027b128d1a45a279acaf'
STATES = {'INSUFFICIENT_INFORMATION': '필수 타일 정보 없음',
          'NEEDS_INPUT_REVIEW': '입력 확인 필요', 'NO_LISTED_ISSUE_DETECTED': '제한 점검 이상 없음'}
RESULTS = {'UNDETERMINED': '판단 불가', 'CONFIRMED': '확인됨', 'NOT_CONFIRMED': '확인되지 않음'}
FIELDS = {'author', 'observation', 'scope_status', 'scope', 'reference', 'result', 'reason', 'change_reason'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Evidence:
    def __init__(self, path: Path):
        # Keep a verified immutable snapshot; later file replacement cannot change a case.
        raw = path.read_bytes()
        self.digest = sha(raw)
        if self.digest != EXPECTED_ZIP_SHA:
            raise ValueError('고정된 전체 원본 점검 ZIP과 다릅니다. 원본을 변경하지 마세요.')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            manifest = json.loads(z.read('handoff-manifest.json'))
            if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {'handoff-manifest.json'}:
                raise ValueError('ZIP 목록이 다릅니다.')
            self.images = {}
            for name, digest in manifest.items():
                content = z.read(name)
                if sha(content) != digest:
                    raise ValueError('첨부 파일 지문이 다릅니다: ' + name)
                if name.startswith('images/') and name.endswith('.png'):
                    self.images[name] = content
            report_bytes = z.read('input-assessment.json')
            self.report = json.loads(report_bytes)
        self.report_sha = sha(report_bytes)
        self.policy_sha = sha(canonical(self.report['policy']).encode())
        if (self.report['mode'] != 'archive' or self.report['production_approved'] is not False
                or self.report['policy']['automatic_pass_allowed'] is not False):
            raise ValueError('연구 입력 점검 계약이 다릅니다.')
        self.cases = {p['group'] + '/' + p['photo_id']: p for p in self.report['photos']}
        if len(self.cases) != 512:
            raise ValueError('512사진 명단이 필요합니다.')

    def image_name(self, case, region):
        return f'images/{case.replace("/", "-")}-{region}.png'


class Conflict(ValueError):
    pass


class Notes:
    def __init__(self, path: Path, evidence: Evidence):
        self.path, self.evidence = path, evidence
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and tables != {'input_review_meta', 'input_review_revisions'}:
                raise ValueError('다른 용도의 DB입니다. 별도 입력 검토 DB를 사용하세요.')
            con.execute('CREATE TABLE IF NOT EXISTS input_review_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            con.execute('''CREATE TABLE IF NOT EXISTS input_review_revisions (
                case_key TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision > 0),
                request_id TEXT NOT NULL UNIQUE, request_json TEXT NOT NULL,
                created_at TEXT NOT NULL, record_json TEXT NOT NULL,
                PRIMARY KEY(case_key, revision))''')
            identity = canonical({'schema': 1, 'bundle_sha256': evidence.digest, 'report_sha256': evidence.report_sha})
            previous = con.execute("SELECT value FROM input_review_meta WHERE key='identity'").fetchone()
            if previous and previous[0] != identity:
                raise ValueError('다른 입력 버전의 DB입니다. 기존 DB를 보존하고 새 경로를 사용하세요.')
            con.execute("INSERT OR IGNORE INTO input_review_meta VALUES ('identity', ?)", (identity,))
            for table in ('input_review_meta', 'input_review_revisions'):
                for operation in ('UPDATE', 'DELETE'):
                    con.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_{operation.lower()} BEFORE {operation} ON {table}
                                    BEGIN SELECT RAISE(ABORT, 'append-only input review'); END''')

    def all_records(self):
        with sqlite3.connect(self.path) as con:
            return [json.loads(r[0]) for r in con.execute(
                'SELECT record_json FROM input_review_revisions ORDER BY case_key, revision')]

    def history(self, case):
        with sqlite3.connect(self.path) as con:
            return [json.loads(r[0]) for r in con.execute(
                'SELECT record_json FROM input_review_revisions WHERE case_key=? ORDER BY revision', (case,))]

    def save(self, case, fields, regions, base, request_id):
        if case not in self.evidence.cases or set(fields) != FIELDS:
            raise ValueError('사진 또는 입력 항목이 다릅니다.')
        if type(base) is not int or base < 0 or len(request_id) != 32 or any(c not in '0123456789abcdef' for c in request_id):
            raise ValueError('저장 요청 정보가 잘못됐습니다.')
        if any(not isinstance(v, str) or len(v) > 2000 for v in fields.values()):
            raise ValueError('각 입력은 2,000자 이하여야 합니다.')
        fields = {k: v.strip() for k, v in fields.items()}
        if not fields['author'] or len(fields['author']) > 80 or not fields['observation'] or not fields['reason']:
            raise ValueError('작성자(80자 이하), 관찰 내용, 판단 이유를 입력하세요.')
        if fields['scope_status'] not in ('UNKNOWN', 'DOCUMENTED') or fields['result'] not in RESULTS:
            raise ValueError('허용되지 않은 판단 값입니다.')
        if fields['scope_status'] == 'UNKNOWN' and fields['result'] != 'UNDETERMINED':
            raise ValueError('필수 검사 범위가 미정이면 판단 불가로 기록하세요.')
        if fields['scope_status'] == 'DOCUMENTED' and (not fields['scope'] or not fields['reference']):
            raise ValueError('검사 범위와 근거 문서·위치를 모두 입력하세요.')
        if base and not fields['change_reason']:
            raise ValueError('수정 이유를 입력하세요. 이전 기록은 보존됩니다.')
        source = {r['region']: r for r in self.evidence.cases[case]['regions']}
        if not regions or len(regions) != len(set(regions)) or set(regions) - source.keys():
            raise ValueError('관련 영역을 하나 이상 선택하세요.')
        if fields['result'] == 'CONFIRMED':
            if any(source[r]['state'] == 'INSUFFICIENT_INFORMATION' for r in regions):
                raise ValueError('전체가 검은 타일 자체의 가시성을 확인됨으로 기록할 수 없습니다.')
            if any(self.evidence.image_name(case, r) not in self.evidence.images for r in regions):
                raise ValueError('원본이 없는 영역은 이 화면에서 가시성 확인됨으로 기록할 수 없습니다.')
        request = canonical({'case': case, 'fields': fields, 'regions': sorted(regions), 'base': base})
        with sqlite3.connect(self.path, timeout=10) as con:
            con.execute('BEGIN IMMEDIATE')
            existing = con.execute('SELECT request_json, record_json FROM input_review_revisions WHERE request_id=?', (request_id,)).fetchone()
            if existing:
                if existing[0] != request:
                    raise Conflict('같은 저장 요청의 내용이 달라졌습니다. 화면을 다시 열어주세요.')
                return json.loads(existing[1])
            current = con.execute('SELECT COALESCE(MAX(revision),0) FROM input_review_revisions WHERE case_key=?', (case,)).fetchone()[0]
            if current != base:
                raise Conflict('다른 창에서 기록이 변경됐습니다. 아래 입력을 보존한 뒤 최신 이력을 다시 열어주세요.')
            record = dict(fields, schema_version=1, case_key=case, revision=current+1,
                          previous_revision=current or None, regions=sorted(regions),
                          created_at=datetime.now(timezone.utc).isoformat(), request_id=request_id,
                          author_kind='SELF_REPORTED_LOCAL_REVIEWER', question='선택 영역에서 기재한 필수 검사 부위가 보이는가?',
                          bundle_sha256=self.evidence.digest, report_sha256=self.evidence.report_sha,
                          policy_sha256=self.evidence.policy_sha,
                          region_sources={r: source[r]['source'] for r in sorted(regions)},
                          automatic_pass_allowed=False, production_approved=False)
            con.execute('INSERT INTO input_review_revisions VALUES (?,?,?,?,?,?)',
                        (case, current+1, request_id, request, record['created_at'], canonical(record)))
            return record


def esc(value):
    return html.escape(str(value), quote=True)


def page(title, body):
    return HTMLResponse(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width"><title>{esc(title)}</title>
    <style>body{{font:16px system-ui;line-height:1.6;max-width:1100px;margin:30px auto;padding:0 20px}}
    .notice{{background:#fff2cc;padding:16px}}.error{{background:#ffdddd;padding:16px}}
    table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:8px;text-align:left}}
    label{{display:block;margin:12px 0}}textarea{{display:block;width:95%;min-height:80px}}
    input[type=text]{{width:80%}}img{{width:256px;max-width:100%;height:auto}}button{{padding:12px;margin:12px 0}}
    pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{margin:12px 0}}</style>
    <h1>{esc(title)}</h1><p class="notice">연구용 입력 검토 · 제품 합격·불량·조치 승인 아님.
    모델 경고와 기존 성적은 변경되지 않습니다. 작성자 이름은 직접 입력하며 신원 인증 기능은 없습니다.</p>{body}</html>''')


def create_app(evidence: Evidence, notes: Notes):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    csrf = secrets.token_hex(32)

    @app.middleware('http')
    async def security(request, call_next):
        # Reject DNS rebinding hosts; the launcher only binds to 127.0.0.1.
        if request.url.hostname not in ('127.0.0.1', 'localhost'):
            return Response('Local access only', status_code=403)
        if request.method == 'POST' and request.headers.get('origin') not in (None, str(request.base_url).rstrip('/')):
            return Response('Cross-origin request rejected', status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.get('/')
    def index():
        latest = {r['case_key']: r for r in notes.all_records()}
        cases = sorted(evidence.cases.items(), key=lambda item: (item[1]['state'] == 'NO_LISTED_ISSUE_DETECTED', item[0]))
        rows = ''.join(f'<tr><td><a href="/case/{k}">{k}</a></td><td>{STATES[p["state"]]}</td>'
                       f'<td>{"미기록" if k not in latest else "의견 r"+str(latest[k]["revision"])}</td></tr>' for k,p in cases)
        # All 512 cases remain visible; priority is not exclusion or a pass list.
        return page('MPI 입력 검토와 의견 기록', f'<p>전체 512장 · 입력 확인 대상 32장을 먼저 표시합니다. '
                    f'의견이 있는 사진 {len(latest)}장. 의견 작성은 입력 적합 승인과 다릅니다.</p>'
                    '<p><a href="/export">전체 의견 이력 JSON 내려받기</a></p>'
                    '<table><tr><th>사진</th><th>입력 관찰</th><th>의견 기록</th></tr>'+rows+'</table>')

    def case_page(key, submitted=None, selected=None, base=None, request_id=None, error=None, saved=False):
        if key not in evidence.cases:
            return Response('Unknown case', status_code=404)
        p = evidence.cases[key]; history = notes.history(key)
        latest = history[-1] if history else {}
        values = submitted if submitted is not None else {k: latest.get(k, '') for k in FIELDS}
        # Never attribute a prior writer's correction to the current reviewer automatically.
        if submitted is None:
            values['author'] = ''; values['change_reason'] = ''
        base = base if base is not None else len(history)
        selected = selected if selected is not None else latest.get('regions', [])
        parts = ['<a href="/">목록으로</a> · <a href="#opinion">의견 작성으로 이동</a>']
        if saved:
            parts.append('<p role="status">의견을 저장했습니다. 모델 판정이나 승인은 바뀌지 않았습니다.</p>')
        if error:
            parts.append(f'<p class="error" role="alert">{esc(error)}</p><a href="/case/{key}">최신 기록 다시 열기</a>')
        parts.append(f'<p>자동 입력 관찰: {STATES[p["state"]]} · 기존 원본 점검의 사람 판단은 별도 보존됩니다.</p>')
        for r in p['regions']:
            region = r['region']; name = evidence.image_name(key, region)
            picture = (f'<img src="/image/{key}/{region}" alt="{key} {region} 원본">' if name in evidence.images
                       else '<p>이 영역 원본은 첨부되지 않았습니다. 통계만으로 가시성을 확정하지 마세요.</p>')
            parts.append(f'<details {"open" if r["state"] != "NO_LISTED_ISSUE_DETECTED" else ""}><summary>{region} · '
                         f'{STATES[r["state"]]} · 검은 픽셀 {r["statistics"]["connected_black_pixels"]:,}</summary>{picture}</details>')
        parts.append(f'<h2 id="opinion">입력 관찰 의견</h2><p>질문: 선택 영역에서 아래에 기재한 필수 검사 부위가 보이나요?</p>'
                     f'<form method="post" action="/case/{key}"><input type="hidden" name="csrf" value="{csrf}">'
                     f'<input type="hidden" name="base" value="{base}"><input type="hidden" name="request_id" value="{esc(request_id or uuid4().hex)}">')
        for r in p['regions']:
            region = r['region']
            parts.append(f'<label><input type="checkbox" name="regions" value="{region}" {"checked" if region in selected else ""}>{region}</label>')
        parts.append(f'<label>작성자 <input type="text" name="author" maxlength="80" required value="{esc(values.get("author", ""))}"></label>')
        for field, label in [('observation','보이는 사실과 관련 위치'), ('scope','필수 검사 부위와 범위'),
                             ('reference','근거 문서와 절·위치 (미정이면 비워 두세요)'),
                             ('reason','판단 근거 또는 판단 불가 이유'), ('change_reason','수정 이유 (개정 시 필수)')]:
            parts.append(f'<label>{label}<textarea name="{field}" maxlength="2000" {"required" if field in ("observation", "reason") else ""}>{esc(values.get(field,""))}</textarea></label>')
        for field, label, options in [('scope_status','검사 범위의 근거',{'UNKNOWN':'미정','DOCUMENTED':'근거 문서 있음 (작성자 제시, 승인 아님)'}),
                                      ('result','선택한 필수 부위의 가시성',RESULTS)]:
            parts.append(f'<label>{label}<select name="{field}">')
            for value, title in options.items():
                parts.append(f'<option value="{value}" {"selected" if values.get(field)==value else ""}>{title}</option>')
            parts.append('</select></label>')
        parts.append('<p>범위가 미정이면 판단 불가를 선택하세요. 전체가 검거나 원본이 없는 선택 영역은 확인됨으로 저장할 수 없습니다.</p>'
                     '<button type="submit">의견 저장 · 합격이나 승인 아님</button></form><h2>수정 이력</h2>')
        for record in reversed(history):
            readable = ''.join(f'<dt>{label}</dt><dd style="white-space:pre-wrap">{esc(record.get(field, "") or "—")}</dd>'
                               for field, label in [('observation','보이는 사실'),('scope','필수 검사 범위'),
                                                    ('reference','근거 문서'),('reason','판단 이유'),('change_reason','수정 이유')])
            parts.append(f'<details><summary>r{record["revision"]} · {esc(record["author"])} · {esc(record["created_at"])} (UTC)</summary>'
                         f'<p>관련 영역: {esc(", ".join(record["regions"]))}<br>검사 범위: '
                         f'{"미정" if record["scope_status"]=="UNKNOWN" else "작성자가 근거 제시"}<br>'
                         f'가시성 의견: {RESULTS[record["result"]]}</p><dl>{readable}</dl>'
                         f'<details><summary>출처와 전체 저장 기록</summary><pre>{esc(json.dumps(record,ensure_ascii=False,indent=2))}</pre></details></details>')
        if not history:
            parts.append('<p>아직 저장된 의견이 없습니다.</p>')
        model = p['model_result']
        text = '모델 결과 없음 (학습자료)' if model is None else ('기존 모델 경고' if model['alert'] else '경고선 미만 · 합격 아님')
        parts.append(f'<details><summary>기존 모델 결과 별도 확인</summary><p>{text}</p>'
                     '<p>불량 55/55 검출 · 시험 정상 0/54 오경보 · 별도 정상 15/256 오경보. 저장된 이력이며 새 추론 아님.</p></details>')
        return page(key+' 입력 검토', ''.join(parts))

    @app.get('/case/{group}/{pid}')
    def get_case(group: str, pid: str, saved: bool = False):
        return case_page(group+'/'+pid, saved=saved)

    @app.post('/case/{group}/{pid}')
    async def post_case(group: str, pid: str, request: Request):
        if request.headers.get('content-type','').split(';')[0] != 'application/x-www-form-urlencoded':
            return Response('Unsupported content type', status_code=415)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 200000:
                return Response('Form too large', status_code=413)
        try:
            form = parse_qs(body.decode('utf-8'), keep_blank_values=True, max_num_fields=40)
            if set(form) - (FIELDS | {'regions','csrf','base','request_id'}) or any(len(v)!=1 for k,v in form.items() if k!='regions'):
                raise ValueError('중복되거나 알 수 없는 입력 항목입니다.')
            if not hmac.compare_digest(form.get('csrf',[''])[0].encode(), csrf.encode()):
                return Response('저장 토큰이 만료됐습니다. 화면을 다시 열어주세요.', status_code=403)
            fields = {k: form[k][0] for k in FIELDS}
            regions = form.get('regions', []); base = int(form['base'][0]); request_id = form['request_id'][0]
        except (ValueError, KeyError, UnicodeError):
            return Response('잘못된 입력 형식입니다.', status_code=422)
        key = group+'/'+pid
        try:
            notes.save(key, fields, regions, base, request_id)
        except ValueError as exc:
            response = case_page(key, fields, regions, base, request_id, str(exc))
            response.status_code = 409 if isinstance(exc, Conflict) else 422
            return response
        except sqlite3.Error:
            response = case_page(key, fields, regions, base, request_id,
                                 'DB 저장을 확인하지 못했습니다. 입력을 보존하고 이력을 확인한 뒤 같은 요청으로 다시 시도하세요.')
            response.status_code = 503
            return response
        return RedirectResponse('/case/'+key+'?saved=true', status_code=303)

    @app.get('/image/{group}/{pid}/{region}')
    def get_image(group: str, pid: str, region: str):
        content = evidence.images.get(evidence.image_name(group+'/'+pid, region))
        return Response(content, media_type='image/png') if content else Response(status_code=404)

    @app.get('/export')
    def export():
        return JSONResponse({'schema_version':1, 'bundle_sha256':evidence.digest,
                             'report_sha256':evidence.report_sha, 'automatic_pass_allowed':False,
                             'production_approved':False, 'records':notes.all_records()},
                            headers={'Content-Disposition':'attachment; filename="mpi-input-review-notes.json"'})

    return app
