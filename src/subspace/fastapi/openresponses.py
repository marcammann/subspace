import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from subspace.core import ModelNotFoundError, Subspace
from subspace.middleware.context import RequestContext
from subspace.models.common import Status
from subspace.models.events import ErrorEvent, ResponseCompletedEvent
from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource
from subspace.streaming.sse import stream_to_sse

if TYPE_CHECKING:
    from subspace.fastapi.mount import SubspaceMount


async def _no_deps() -> None:
    return None


class OpenResponsesInterface:
    def __init__(self, *, prefix: str = "/v1") -> None:
        self._prefix = prefix

    def build_router(self, mount: "SubspaceMount") -> APIRouter:
        router = APIRouter(prefix=self._prefix)
        subspace = mount.subspace
        interface_mw = list(mount.middlewares)
        deps_getter = mount.deps or _no_deps
        context_class = mount.context_class

        @router.post("/responses")
        async def create_response(
            request: Request,
            body: CreateResponseRequest,
            deps: Any = Depends(deps_getter),
        ):
            try:
                chain = subspace.build_chain(body.model, interface_mw)
            except ModelNotFoundError:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": f"Model not found: {body.model}",
                            "type": "not_found",
                            "code": "model_not_found",
                        }
                    },
                )

            response_id = f"resp_{uuid.uuid4().hex[:24]}"
            ctx = context_class(
                request=body,
                response_id=response_id,
                response=ResponseResource(
                    id=response_id,
                    created_at=int(time.time()),
                    status=Status.IN_PROGRESS,
                    model=body.model,
                    instructions=body.instructions,
                    tools=body.tools,
                ),
                app=request.app,
                deps=deps,
            )

            if body.stream:
                return EventSourceResponse(
                    _stream_response(chain, ctx),
                    media_type="text/event-stream",
                )

            return await _non_stream_response(chain, ctx)

        @router.get("/models")
        async def list_models():
            models = subspace.list_models()
            return {
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "owned_by": "subspace"} for name in models
                ],
            }

        return router


async def _stream_response(chain, ctx):
    try:
        async for sse_dict in stream_to_sse(chain.execute(ctx)):
            yield sse_dict
    except Exception as e:
        error = ErrorEvent(
            sequence_number=0,
            code="server_error",
            message=str(e),
        )
        yield {"event": error.type, "data": error.model_dump_json()}
        yield {"data": "[DONE]"}


async def _non_stream_response(chain, ctx):
    final = None
    async for event in chain.execute(ctx):
        if isinstance(event, ResponseCompletedEvent):
            final = event.response

    if final is None:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "No completed response from backend",
                    "type": "server_error",
                    "code": "server_error",
                }
            },
        )

    return JSONResponse(content=final.model_dump(mode="json"))
