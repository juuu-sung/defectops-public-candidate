"""Serve the 3D workspace and forward its inspection requests to one DefectOps API."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Path as PathParam, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware


ROOT = Path(__file__).resolve().parent
COMPARISON_FIGURES = ROOT.parents[1] / "artifacts" / "analyses" / "inspection-map-comparisons"
GROUND_TRUTH_CASES = ROOT.parents[1] / "artifacts" / "analyses" / "inspection-ground-truth"
THRESHOLD_TRIALS = ROOT.parents[1] / "artifacts" / "analyses" / "inspection-threshold-trials"
MAX_PNG = 50 * 1024 * 1024
InspectionId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")]
TrialId = Annotated[str, PathParam(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
Picture = Literal["input.png", "heatmap.png", "overlay.png"]
GroundTruthFile = Literal["case.json", "input-with-mask.png", "heatmap-with-mask.png", "official-mask.png"]


def create_lab_app(api_url: str = "http://127.0.0.1:8000", *, transport=None) -> FastAPI:
    base = httpx.URL(api_url)
    if base.scheme not in {"http", "https"} or not base.host or base.userinfo or base.query or base.fragment:
        raise ValueError("Supply the base HTTP(S) address of the inspection API")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            base_url=api_url.rstrip("/") + "/", transport=transport, trust_env=False,
            timeout=httpx.Timeout(180, connect=4), follow_redirects=False,
        ) as upstream:
            app.state.upstream = upstream
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    async def forward(path: str, content: bytes | None = None) -> Response:
        client = app.state.upstream
        try:
            if content is None:
                result = await client.get(path)
            else:
                result = await client.post(path, content=content, headers={"Content-Type": "image/png"})
        except httpx.TimeoutException as exc:
            raise HTTPException(504, "검사 서버 응답 시간이 초과됐습니다.") from exc
        except httpx.RequestError as exc:
            raise HTTPException(503, "PatchCore 검사 서버에 연결할 수 없습니다.") from exc
        media_type = result.headers.get("content-type", "").partition(";")[0].lower()
        if media_type not in {"application/json", "image/png"} or result.is_redirect:
            raise HTTPException(502, "검사 API의 응답 형식을 확인하세요.")
        return Response(result.content, status_code=result.status_code, media_type=media_type,
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/healthz")
    async def health():
        return await forward("healthz")

    @app.post("/api/v1/inspections")
    async def inspect(request: Request):
        if request.headers.get("content-type", "").partition(";")[0].strip().lower() != "image/png":
            raise HTTPException(415, "RGB PNG 파일을 선택하세요.")
        size = request.headers.get("content-length", "")
        if size.isdecimal() and int(size) > MAX_PNG:
            raise HTTPException(413, "PNG 파일은 50 MiB 이하여야 합니다.")
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_PNG:
                raise HTTPException(413, "PNG 파일은 50 MiB 이하여야 합니다.")
            data.extend(chunk)
        return await forward("v1/inspections", bytes(data))

    @app.get("/api/v1/inspections/{inspection_id}")
    async def record(inspection_id: InspectionId):
        return await forward(f"v1/inspections/{inspection_id}")

    @app.get("/api/v1/inspections/{inspection_id}/artifacts/{filename}")
    async def artifact(inspection_id: InspectionId, filename: Picture):
        return await forward(f"v1/inspections/{inspection_id}/artifacts/{filename}")

    @app.get("/api/v1/inspections/{inspection_id}/gpu-comparison.png")
    async def gpu_comparison(inspection_id: InspectionId):
        figure = COMPARISON_FIGURES / f"{inspection_id}.png"
        if figure.is_symlink() or not figure.is_file():
            raise HTTPException(404, "이 검사에 저장된 GPU 비교 지도가 없습니다.")
        return FileResponse(figure, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/inspections/{inspection_id}/ground-truth/{filename}")
    async def ground_truth(inspection_id: InspectionId, filename: GroundTruthFile):
        case = GROUND_TRUTH_CASES / inspection_id
        file = case / filename
        if case.is_symlink() or file.is_symlink() or not file.is_file():
            raise HTTPException(404, "이 검사에 연결된 정답 마스크 자료가 없습니다.")
        media_type = "application/json" if filename == "case.json" else "image/png"
        return FileResponse(file, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.get("/api/threshold-trials/{trial_id}")
    async def threshold_trial(trial_id: TrialId):
        directory = THRESHOLD_TRIALS / trial_id
        report = directory / "trial.json"
        if directory.is_symlink() or report.is_symlink() or not report.is_file():
            raise HTTPException(404, "저장된 사진 경고 기준 실험을 찾을 수 없습니다.")
        return FileResponse(report, media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get("/api/threshold-trials/{trial_id}/areas/{inspection_id}.png")
    async def threshold_area(trial_id: TrialId, inspection_id: InspectionId):
        directory = THRESHOLD_TRIALS / trial_id
        areas = directory / "areas"
        picture = areas / f"{inspection_id}.png"
        if directory.is_symlink() or areas.is_symlink() or picture.is_symlink() or not picture.is_file():
            raise HTTPException(404, "이 사진의 이상 영역 비교 자료를 찾을 수 없습니다.")
        return FileResponse(picture, media_type="image/png", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=ROOT, html=True), name="workspace")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--port", default=8791, type=int)
    args = parser.parse_args()
    uvicorn.run(create_lab_app(args.api_url), host="127.0.0.1", port=args.port)
