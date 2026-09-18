"""FastAPI pipeline with independent validation before publishing JSON."""

from __future__ import annotations

import asyncio
from importlib import import_module
from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from app.response import build_response
from app.validator import ReplayValidationError, validate_response

from app.errors import PipelineNotImplementedError, ServiceError, error_payload
from app.schemas import (
    DirectiveInterpretation,
    HealthResponse,
    HourlyPlan,
    OptimizeRequest,
    OptimizeResponse,
)

app = FastAPI(title="GridWise LLM", version="0.1.0")
REQUEST_TIMEOUT_SECONDS = 27.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


def _component(module: str, name: str):
    try:
        return getattr(import_module(module), name)
    except ModuleNotFoundError as exc:
        if exc.name == module:
            raise PipelineNotImplementedError() from exc
        raise


# Lazy adapters preserve startup/health while teammate modules are unavailable.
async def interpret_notes(
    request: OptimizeRequest,
) -> object:
    return await _component("app.llm_interpreter", "interpret_notes")(request)


def validate_and_normalize(
    raw: object,
    request: OptimizeRequest,
) -> list[DirectiveInterpretation]:
    return _component("app.guardrails", "validate_and_normalize")(raw, request)


def optimize(
    request: OptimizeRequest,
    interpretations: Sequence[DirectiveInterpretation],
) -> list[HourlyPlan]:
    return _component("app.optimizer", "optimize")(request, interpretations)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "location": [str(part) for part in error["loc"]],
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content=error_payload(
            "invalid_request", "Request validation failed.", details=details
        ),
    )


@app.exception_handler(ServiceError)
async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
    # Never expose arbitrary messages/codes supplied by provider exceptions.
    safe_type = type(exc) if type(exc) in (PipelineNotImplementedError, ReplayValidationError) else ServiceError
    return JSONResponse(
        status_code=500,
        content=error_payload(safe_type.code, safe_type.public_message),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_payload(
            "internal_error", "The service could not complete the request."
        ),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def judge_ui() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/sample-cases", include_in_schema=False)
async def sample_cases() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: OptimizeRequest) -> Response:
    """Return only the exact serialized bytes that passed independent replay."""
    # Thread work can outlive cancellation, but cannot publish a late response.
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            raw = await interpret_notes(request.model_copy(deep=True))
            directives = await asyncio.to_thread(
                validate_and_normalize, raw, request.model_copy(deep=True)
            )
            directives = [d.model_copy(deep=True) for d in directives]
            rows = await asyncio.to_thread(
                optimize, request.model_copy(deep=True),
                [d.model_copy(deep=True) for d in directives],
            )
            return await asyncio.to_thread(_finish_response, request, directives, rows)
    except (PipelineNotImplementedError, ReplayValidationError):
        raise
    except Exception:
        raise ServiceError() from None


def _finish_response(request, directives, rows) -> Response:
    result = build_response(request, directives, rows)
    serialized = result.model_dump_json()
    final = OptimizeResponse.model_validate_json(serialized)
    validate_response(request, final, directives)
    return Response(content=serialized, media_type="application/json")
