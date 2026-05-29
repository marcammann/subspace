from typing import Any

from fastapi.responses import JSONResponse

from subspace.models.common import ResponseError
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
)
from subspace.models.response import ResponseResource


async def no_deps() -> None:
    """Default dependency hook for mounts without dependency injection."""
    return None


async def collect_terminal_response(chain: Any, ctx: Any) -> ResponseResource | None:
    """Consume a chain and return its terminal response resource, if any."""
    final = None
    async for event in chain.execute(ctx):
        if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent)):
            final = event.response
    return final


def openai_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    code: str,
) -> JSONResponse:
    """Return an OpenAI-shaped JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def anthropic_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
) -> JSONResponse:
    """Return an Anthropic-shaped JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
    )


def openai_failed_response(response: ResponseResource) -> JSONResponse:
    """Return an OpenAI-shaped error for a failed internal response."""
    error = response.error or ResponseError(
        message="Backend response failed",
        type="server_error",
        code="server_error",
    )
    return openai_error_response(
        status_code=502,
        message=error.message,
        error_type=error.type,
        code=error.code or "server_error",
    )


def anthropic_failed_response(response: ResponseResource) -> JSONResponse:
    """Return an Anthropic-shaped error for a failed internal response."""
    error = response.error or ResponseError(
        message="Backend response failed",
        type="server_error",
        code="server_error",
    )
    return anthropic_error_response(
        status_code=502,
        message=error.message,
        error_type=error.type,
    )
