"""One error shape for the integration routes (2026-10-21).

``/sessions``, ``/materials`` and ``/catalog`` answered their own refusals as
``{"detail": {"code", "message"}}``, but a wrong password as
``{"detail": "Unauthorized"}`` and a malformed body as FastAPI's
``{"detail": [...]}``: a client had to test whether ``detail`` was an object, a
string or a list. On these routes every error now has ``code`` and ``message``;
a malformed request adds the field errors under ``errors``.

``/context`` and ``/agent/*`` keep their established shapes: they have callers.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

#: The routes whose every error is ``{"detail": {"code", "message"}}``.
ENVELOPE_PREFIXES = ("/api/v1/sessions", "/api/v1/materials", "/api/v1/catalog")

#: The code and Italian message of an error raised without its own code.
_PLAIN = {
    401: ("unauthorized", "Credenziali mancanti o non valide."),
    403: ("forbidden", "Accesso non consentito."),
    404: ("not_found", "Risorsa non trovata."),
    405: ("method_not_allowed", "Metodo non ammesso su questo indirizzo."),
}


class ErrorBody(BaseModel):
    code: str = Field(..., description="Stable, for your code: branch on it.")
    message: str = Field(..., description="In Italian, for a person.")
    errors: Optional[list[dict[str, Any]]] = Field(
        None, description="invalid_request only: each field that is wrong (loc, msg, type)."
    )


class ErrorResponse(BaseModel):
    """Every error of ``/sessions``, ``/materials`` and ``/catalog``."""

    detail: ErrorBody


#: For the routers' ``responses``: what an error looks like in the contract.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse, "description": description}
    for status, description in (
        (401, "Missing or wrong credentials (code: unauthorized)."),
        (422, "The request is malformed (code: invalid_request, with errors) or refused."),
    )
}


def _enveloped(request: Request) -> bool:
    return request.url.path.startswith(ENVELOPE_PREFIXES)


async def _http_error(request: Request, exc: StarletteHTTPException):
    if not _enveloped(request) or isinstance(exc.detail, dict):
        return await http_exception_handler(request, exc)
    code, message = _PLAIN.get(exc.status_code, ("http_error", str(exc.detail)))
    return JSONResponse(
        {"detail": {"code": code, "message": message}},
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def _invalid_request(request: Request, exc: RequestValidationError):
    if not _enveloped(request):
        return await request_validation_exception_handler(request, exc)
    errors = jsonable_encoder(exc.errors())
    first = errors[0] if errors else {}
    where = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = "La richiesta non è valida" + (f": {where}" if where else "") + "."
    return JSONResponse(
        {"detail": {"code": "invalid_request", "message": message, "errors": errors}},
        status_code=422,
    )


def install(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _invalid_request)
