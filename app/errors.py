"""Controlled service errors and safe public error payloads."""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    status_code = 500
    code = "internal_error"
    public_message = "The service could not complete the request."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)


class PipelineNotImplementedError(ServiceError):
    code = "pipeline_not_implemented"
    public_message = "The optimization pipeline is not implemented yet."


def error_payload(
    code: str,
    message: str,
    *,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}

