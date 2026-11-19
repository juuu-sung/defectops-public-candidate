"""Local-only PatchCore HTTP inspection service with one resident model."""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from time import perf_counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from defectops.padim_experiment import checksum, read_json, write_json
from defectops.padim_inspection import (
    InputValidationError,
    PadimRuntime,
    _safe_named_file,
    load_inference_runtime,
    run_inspection,
    validate_expected_model,
)
from defectops.review_registration import ReviewRegistration


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ARTIFACT_CONTENT_TYPES = {
    "input.png": "image/png",
    "heatmap.png": "image/png",
    "overlay.png": "image/png",
    "anomaly-map.npz": "application/octet-stream",
}
ARTIFACT_KEYS = {
    "input.png": "input",
    "heatmap.png": "heatmap",
    "overlay.png": "overlay",
    "anomaly-map.npz": "anomaly_map",
}


def _failure_record(output_path: Path, error_type: str, message: str) -> dict[str, Any]:
    """Persist an API-boundary failure without ever emitting a PASS decision."""
    now = datetime.now(timezone.utc).isoformat()
    output_path.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "status": "INSPECTION_FAILED",
        "inspection_id": output_path.name,
        "started_at": now,
        "finished_at": now,
        "input_path": "api-upload",
        "error_type": error_type,
        "error": message,
        "automatic_pass": False,
        "automatic_pass_allowed": False,
    }
    write_json(output_path / "inspection.json", record)
    return record


def _inspection_directory(root: Path, inspection_id: str) -> Path:
    """Resolve only the server-generated UUID hex names below the artifact root."""
    try:
        if UUID(inspection_id).hex != inspection_id:
            raise ValueError("non-canonical inspection ID")
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=404, detail="Inspection not found") from exc
    directory = (root / inspection_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Inspection not found") from exc
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Inspection not found")
    return directory


def create_app(
    fit_run: Path,
    evaluation: Path,
    *,
    output_root: Path | None = None,
    review_database: Path | None = None,
    runtime_loader: Callable[[Path, Path, Path], PadimRuntime] = load_inference_runtime,
    inspection_runner: Callable[..., dict[str, Any]] = run_inspection,
) -> FastAPI:
    """Create a local inspection app; load and validate the model only at startup."""
    project_root = Path(__file__).resolve().parents[2]
    artifact_root = (output_root or project_root / "artifacts" / "inspections").resolve()
    fit_path = fit_run.resolve()
    evaluation_path = evaluation.resolve()
    registration = ReviewRegistration(artifact_root, review_database) if review_database is not None else None
    if output_root is None:
        fit_path.relative_to((project_root / "artifacts" / "runs").resolve())
        evaluation_path.relative_to((project_root / "artifacts" / "evaluations").resolve())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        artifact_root.mkdir(parents=True, exist_ok=True)
        if registration is not None:
            # Fail startup before loading the GPU model if the configured DB cannot be opened.
            await run_in_threadpool(registration.initialize)
            app.state.review_recovery = await run_in_threadpool(registration.reconcile)
        resolved_fit = fit_path.resolve(strict=True)
        resolved_evaluation = evaluation_path.resolve(strict=True)
        runtime = runtime_loader(project_root, resolved_fit, resolved_evaluation)
        validate_expected_model(runtime.fit_config["model"], "patchcore")
        app.state.runtime = runtime
        app.state.inspection_lock = asyncio.Lock()
        try:
            yield
        finally:
            app.state.runtime = None

    app = FastAPI(title="DefectOps local PatchCore inspection", lifespan=lifespan)

    async def finish(record: dict, status_code: int) -> JSONResponse:
        response = dict(record)
        if registration is not None:
            start = perf_counter()
            state = await run_in_threadpool(registration.inspect, record['inspection_id'], register=True)
            response['review_registration'] = dict(state, elapsed_ms=(perf_counter()-start)*1000)
        return JSONResponse(status_code=status_code, content=response)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        runtime: PadimRuntime = request.app.state.runtime
        device = runtime.device.type
        return {
            "status": "ready",
            "git_provenance": getattr(runtime, "git_provenance", {}),
            "model": "patchcore",
            "device": device,
            "comparison_status": runtime.fit_run.get("comparison_status"),
            "fit_run_id": runtime.fit_run["run_id"],
            "process_id": os.getpid(),
            "model_setup_ms": runtime.setup_ms,
            "gpu_memory": {
                "allocated_bytes": runtime.torch.cuda.memory_allocated(),
                "reserved_bytes": runtime.torch.cuda.memory_reserved(),
            } if device == "cuda" else None,
            "automatic_pass_allowed": False,
            "review_registration": {
                "enabled": registration is not None,
                "startup_recovery": getattr(request.app.state, 'review_recovery', None),
            },
        }

    @app.post("/v1/inspections", status_code=201)
    async def inspect(request: Request) -> JSONResponse:
        inspection_id = uuid4().hex
        output_path = artifact_root / inspection_id
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "image/png":
            failure = _failure_record(
                output_path, "UnsupportedMediaType", "Expected Content-Type: image/png"
            )
            return await finish(failure, 415)
        declared_size = request.headers.get("content-length", "")
        if declared_size.isdecimal() and int(declared_size) > MAX_UPLOAD_BYTES:
            failure = _failure_record(
                output_path, "InputTooLarge", "Input PNG exceeds the 50 MiB safety limit"
            )
            return await finish(failure, 413)

        with tempfile.TemporaryDirectory(prefix="defectops-api-") as temp_dir:
            input_path = Path(temp_dir) / "upload.png"
            size = 0
            try:
                with input_path.open("xb") as stream:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            failure = _failure_record(
                                output_path,
                                "InputTooLarge",
                                "Input PNG exceeds the 50 MiB safety limit",
                            )
                            return await finish(failure, 413)
                        stream.write(chunk)
            except Exception as exc:
                if output_path.exists():
                    raise
                failure = _failure_record(output_path, type(exc).__name__, str(exc))
                return await finish(failure, 400)

            args = argparse.Namespace(
                input=input_path,
                fit_run=fit_path,
                evaluation=evaluation_path,
                output=output_path,
            )
            try:
                async with request.app.state.inspection_lock:
                    result = await run_in_threadpool(
                        inspection_runner,
                        args,
                        expected_model="patchcore",
                        runtime=request.app.state.runtime,
                        input_source="api-upload",
                    )
            except InputValidationError:
                failure = read_json(output_path / "inspection.json")
                return await finish(failure, 422)
            except Exception:
                # The inspection runner writes a detailed failure record to disk.
                return await finish(
                    {
                        "status": "INSPECTION_FAILED",
                        "inspection_id": inspection_id,
                        "automatic_pass": False,
                        "automatic_pass_allowed": False,
                    }, 500,
                )
        return await finish(result, 201)

    @app.get("/v1/inspections/{inspection_id}")
    async def get_inspection(inspection_id: str) -> dict[str, Any]:
        directory = _inspection_directory(artifact_root, inspection_id)
        try:
            path = _safe_named_file(directory, "inspection.json")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Inspection not found") from exc
        result = read_json(path)
        if registration is not None:
            result['review_registration'] = await run_in_threadpool(registration.inspect, inspection_id)
        return result

    @app.post("/v1/inspections/{inspection_id}/review-registration")
    async def retry_review_registration(inspection_id: str) -> JSONResponse:
        if registration is None:
            raise HTTPException(status_code=409, detail="Review registration is disabled")
        _inspection_directory(artifact_root, inspection_id)
        state = await run_in_threadpool(registration.inspect, inspection_id, register=True)
        code = {'REGISTERED': 200, 'PENDING': 503, 'BLOCKED': 409}[state['status']]
        return JSONResponse(status_code=code, content=state)

    @app.get("/v1/inspections/{inspection_id}/artifacts/{filename}")
    async def get_artifact(inspection_id: str, filename: str) -> FileResponse:
        if filename not in ARTIFACT_CONTENT_TYPES:
            raise HTTPException(status_code=404, detail="Artifact not found")
        directory = _inspection_directory(artifact_root, inspection_id)
        try:
            record_path = _safe_named_file(directory, "inspection.json")
            artifact_path = _safe_named_file(directory, filename)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        record = read_json(record_path)
        key = ARTIFACT_KEYS[filename]
        expected_checksum = record.get("artifacts", {}).get(f"{key}_sha256")
        if record.get("status") != "completed" or checksum(artifact_path) != expected_checksum:
            raise HTTPException(status_code=409, detail="Artifact integrity check failed")
        return FileResponse(artifact_path, media_type=ARTIFACT_CONTENT_TYPES[filename])

    return app
