import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from subspace.core import AgentNotFoundError
from subspace.models.common import Role, Status, Usage
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ErrorEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
)
from subspace.models.items import FunctionCall, FunctionCallOutput, InputMessage, OutputMessage
from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource
from subspace.models.tools import FunctionTool, ToolChoiceFunction

from ._shared import (
    anthropic_error_response,
    anthropic_failed_response,
    collect_terminal_response,
    no_deps,
)

if TYPE_CHECKING:
    from subspace.fastapi.mount import SubspaceMount


# ---------------------------------------------------------------------------
# Anthropic wire-format request models
# ---------------------------------------------------------------------------


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    stream: bool = False
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Inbound translation
# ---------------------------------------------------------------------------


def _to_internal_request(body: AnthropicMessageRequest) -> CreateResponseRequest:
    instructions: str | None = None
    if isinstance(body.system, str):
        instructions = body.system
    elif isinstance(body.system, list):
        instructions = (
            "\n".join(block["text"] for block in body.system if block.get("type") == "text") or None
        )

    items: list[Any] = []

    for msg in body.messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                items.append(InputMessage(role=Role.USER, content=msg.content))
            else:
                text_parts: list[str] = []
                for block in msg.content:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_result":
                        if text_parts:
                            items.append(InputMessage(role=Role.USER, content=" ".join(text_parts)))
                            text_parts = []
                        output = block.get("content", "")
                        if isinstance(output, list):
                            output = " ".join(
                                b.get("text", "") for b in output if b.get("type") == "text"
                            )
                        items.append(
                            FunctionCallOutput(
                                call_id=block["tool_use_id"],
                                output=str(output),
                            )
                        )
                if text_parts:
                    items.append(InputMessage(role=Role.USER, content=" ".join(text_parts)))

        elif msg.role == "assistant":
            if isinstance(msg.content, str):
                items.append(InputMessage(role=Role.ASSISTANT, content=msg.content))
            else:
                text_parts = []
                for block in msg.content:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        if text_parts:
                            items.append(
                                InputMessage(
                                    role=Role.ASSISTANT,
                                    content=" ".join(text_parts),
                                )
                            )
                            text_parts = []
                        items.append(
                            FunctionCall(
                                id=f"fc_{uuid.uuid4().hex[:24]}",
                                call_id=block["id"],
                                name=block["name"],
                                arguments=json.dumps(block.get("input", {})),
                                status=Status.COMPLETED,
                            )
                        )
                if text_parts:
                    items.append(InputMessage(role=Role.ASSISTANT, content=" ".join(text_parts)))

    tools: list[FunctionTool] | None = None
    if body.tools:
        tools = [
            FunctionTool(name=t.name, description=t.description, parameters=t.input_schema)
            for t in body.tools
        ]

    tool_choice: str | ToolChoiceFunction | None = None
    if body.tool_choice is not None:
        tc_type = body.tool_choice.get("type")
        if tc_type == "auto":
            tool_choice = "auto"
        elif tc_type == "any":
            tool_choice = "required"
        elif tc_type == "tool":
            tool_choice = ToolChoiceFunction(name=body.tool_choice["name"])

    return CreateResponseRequest(
        model=body.model,
        input=items,
        instructions=instructions,
        tools=tools,
        tool_choice=tool_choice,
        temperature=body.temperature,
        top_p=body.top_p,
        max_output_tokens=body.max_tokens,
        stream=body.stream,
        metadata=body.metadata,
    )


# ---------------------------------------------------------------------------
# Outbound streaming
# ---------------------------------------------------------------------------


async def _stream_response(chain, ctx, msg_id: str, model: str):
    content_index = 0
    # Maps output_index → content_index for tracking which block we're in
    output_to_content: dict[int, int] = {}
    has_tool_use = False

    try:
        async for event in chain.execute(ctx):
            if isinstance(event, ResponseCreatedEvent):
                yield {
                    "event": "message_start",
                    "data": json.dumps(
                        {
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model,
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        }
                    ),
                }

            elif isinstance(event, ResponseOutputItemAddedEvent):
                if isinstance(event.item, OutputMessage):
                    idx = content_index
                    output_to_content[event.output_index] = idx
                    content_index += 1
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {"type": "text", "text": ""},
                            }
                        ),
                    }
                elif isinstance(event.item, FunctionCall):
                    has_tool_use = True
                    idx = content_index
                    output_to_content[event.output_index] = idx
                    content_index += 1
                    yield {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": event.item.call_id,
                                    "name": event.item.name,
                                    "input": {},
                                },
                            }
                        ),
                    }

            elif isinstance(event, ResponseOutputTextDeltaEvent):
                idx = output_to_content.get(event.output_index, 0)
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": event.delta},
                        }
                    ),
                }

            elif isinstance(event, ResponseFunctionCallArgsDeltaEvent):
                idx = output_to_content.get(event.output_index, 0)
                yield {
                    "event": "content_block_delta",
                    "data": json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": event.delta,
                            },
                        }
                    ),
                }

            elif isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
                event.item, (OutputMessage, FunctionCall)
            ):
                if isinstance(event.item, FunctionCall):
                    has_tool_use = True
                idx = output_to_content.get(event.output_index, 0)
                yield {
                    "event": "content_block_stop",
                    "data": json.dumps(
                        {
                            "type": "content_block_stop",
                            "index": idx,
                        }
                    ),
                }

            elif isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                stop_reason = "tool_use" if has_tool_use else "end_turn"
                output_tokens = 0
                if event.response.usage:
                    output_tokens = event.response.usage.output_tokens
                yield {
                    "event": "message_delta",
                    "data": json.dumps(
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason,
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": output_tokens},
                        }
                    ),
                }
                yield {
                    "event": "message_stop",
                    "data": json.dumps({"type": "message_stop"}),
                }

            elif isinstance(event, ResponseFailedEvent):
                error = event.response.error
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": error.type if error else "server_error",
                                "message": error.message if error else "Backend response failed",
                            },
                        }
                    ),
                }

            elif isinstance(event, ErrorEvent):
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": "server_error",
                                "message": event.message,
                            },
                        }
                    ),
                }

    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "type": "error",
                    "error": {"type": "server_error", "message": str(e)},
                }
            ),
        }


# ---------------------------------------------------------------------------
# Outbound non-streaming
# ---------------------------------------------------------------------------


async def _non_stream_response(chain, ctx, msg_id: str, model: str):
    final = await collect_terminal_response(chain, ctx)

    if final is None:
        return anthropic_error_response(
            status_code=502,
            message="No terminal response from backend",
            error_type="server_error",
        )
    if final.status is Status.FAILED:
        return anthropic_failed_response(final)

    content: list[dict[str, Any]] = []
    has_tool_use = False

    for item in final.output:
        if isinstance(item, OutputMessage):
            for part in item.content:
                if isinstance(part, OutputTextContent):
                    content.append({"type": "text", "text": part.text})
        elif isinstance(item, FunctionCall):
            has_tool_use = True
            try:
                input_data = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                input_data = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": item.call_id,
                    "name": item.name,
                    "input": input_data,
                }
            )

    stop_reason = "tool_use" if has_tool_use else "end_turn"
    usage = final.usage or Usage()

    return JSONResponse(
        content={
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        }
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class AnthropicMessagesRouter:
    """Router for the Anthropic Messages-compatible interface."""

    def __init__(self, *, prefix: str = "/v1") -> None:
        self._prefix = prefix

    def build_router(self, mount: "SubspaceMount") -> APIRouter:
        router = APIRouter(prefix=self._prefix)
        subspace = mount.subspace
        interface_mw = list(mount.middlewares)
        deps_getter = mount.deps or no_deps
        context_class = mount.context_class

        deps_dependency = Depends(deps_getter)

        @router.post("/messages")
        async def create_message(
            request: Request,
            body: AnthropicMessageRequest,
            deps: Any = deps_dependency,
        ):
            internal_req = _to_internal_request(body)
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"

            try:
                chain = subspace.build_chain(internal_req.model, interface_mw)
            except AgentNotFoundError:
                return anthropic_error_response(
                    status_code=404,
                    message=f"Model not found: {internal_req.model}",
                    error_type="not_found_error",
                )

            ctx = context_class(
                request=internal_req,
                response_id=msg_id,
                response=ResponseResource(
                    id=msg_id,
                    created_at=int(time.time()),
                    status=Status.IN_PROGRESS,
                    model=internal_req.model,
                    instructions=internal_req.instructions,
                    tools=internal_req.tools,
                ),
                app=request.app,
                deps=deps,
            )

            if body.stream:
                return EventSourceResponse(
                    _stream_response(chain, ctx, msg_id, body.model),
                    media_type="text/event-stream",
                )

            return await _non_stream_response(chain, ctx, msg_id, body.model)

        return router
