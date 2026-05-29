import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from subspace.core import AgentNotFoundError
from subspace.models.common import Role, Status, Usage
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputTextDeltaEvent,
)
from subspace.models.items import FunctionCall, FunctionCallOutput, InputMessage, OutputMessage
from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource
from subspace.models.tools import FunctionTool, ToolChoiceFunction

from ._shared import (
    collect_terminal_response,
    no_deps,
    openai_error_response,
    openai_failed_response,
)

if TYPE_CHECKING:
    from subspace.fastapi.mount import SubspaceMount


# ---------------------------------------------------------------------------
# Chat Completion request models
# ---------------------------------------------------------------------------


class ChatCompletionToolCallFunction(BaseModel):
    name: str
    arguments: str


class ChatCompletionToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ChatCompletionToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ChatCompletionToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ChatCompletionTool(BaseModel):
    type: Literal["function"] = "function"
    function: ChatCompletionToolFunction


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    tools: list[ChatCompletionTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    n: int | None = 1


# ---------------------------------------------------------------------------
# Inbound translation
# ---------------------------------------------------------------------------


def _to_internal_request(body: ChatCompletionRequest) -> CreateResponseRequest:
    instructions_parts: list[str] = []
    items: list[Any] = []

    for msg in body.messages:
        if msg.role == "system" or msg.role == "developer":
            if isinstance(msg.content, str):
                instructions_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        instructions_parts.append(part["text"])

        elif msg.role == "user":
            items.append(InputMessage(role=Role.USER, content=msg.content or ""))

        elif msg.role == "assistant":
            if msg.content:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                items.append(InputMessage(role=Role.ASSISTANT, content=text))
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    items.append(
                        FunctionCall(
                            id=f"fc_{uuid.uuid4().hex[:24]}",
                            call_id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                            status=Status.COMPLETED,
                        )
                    )

        elif msg.role == "tool":
            items.append(
                FunctionCallOutput(
                    call_id=msg.tool_call_id or "",
                    output=msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                )
            )

    instructions = "\n".join(instructions_parts) if instructions_parts else None

    tools: list[FunctionTool] | None = None
    if body.tools:
        tools = [
            FunctionTool(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,
                strict=t.function.strict,
            )
            for t in body.tools
        ]

    tool_choice: str | ToolChoiceFunction | None = None
    if isinstance(body.tool_choice, str):
        tool_choice = body.tool_choice
    elif isinstance(body.tool_choice, dict):
        if body.tool_choice.get("type") == "function" and "function" in body.tool_choice:
            tool_choice = ToolChoiceFunction(name=body.tool_choice["function"]["name"])
        else:
            tool_choice = body.tool_choice.get("type")

    max_output_tokens = body.max_completion_tokens or body.max_tokens

    return CreateResponseRequest(
        model=body.model,
        input=items,
        instructions=instructions,
        tools=tools,
        tool_choice=tool_choice,
        temperature=body.temperature,
        top_p=body.top_p,
        max_output_tokens=max_output_tokens,
        parallel_tool_calls=body.parallel_tool_calls,
        stream=body.stream,
        user=body.user,
    )


# ---------------------------------------------------------------------------
# Outbound streaming
# ---------------------------------------------------------------------------


def _make_chunk(
    chat_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def _stream_chat_response(chain, ctx, chat_id: str, model: str, created: int):
    tool_call_index: dict[str, int] = {}
    next_tool_index = 0

    try:
        async for event in chain.execute(ctx):
            if isinstance(event, ResponseCreatedEvent):
                chunk = _make_chunk(chat_id, model, created, {"role": "assistant", "content": ""})
                yield {"data": json.dumps(chunk)}

            elif isinstance(event, ResponseOutputTextDeltaEvent):
                chunk = _make_chunk(chat_id, model, created, {"content": event.delta})
                yield {"data": json.dumps(chunk)}

            elif isinstance(event, ResponseOutputItemAddedEvent) and isinstance(
                event.item, FunctionCall
            ):
                idx = next_tool_index
                tool_call_index[event.item.call_id] = idx
                next_tool_index += 1
                delta = {
                    "tool_calls": [
                        {
                            "index": idx,
                            "id": event.item.call_id,
                            "type": "function",
                            "function": {
                                "name": event.item.name,
                                "arguments": "",
                            },
                        }
                    ]
                }
                chunk = _make_chunk(chat_id, model, created, delta)
                yield {"data": json.dumps(chunk)}

            elif isinstance(event, ResponseFunctionCallArgsDeltaEvent):
                idx = tool_call_index.get(event.call_id, 0)
                delta = {
                    "tool_calls": [
                        {
                            "index": idx,
                            "function": {"arguments": event.delta},
                        }
                    ]
                }
                chunk = _make_chunk(chat_id, model, created, delta)
                yield {"data": json.dumps(chunk)}

            elif isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                has_tool_calls = any(
                    isinstance(item, FunctionCall) for item in event.response.output
                )
                finish_reason = "tool_calls" if has_tool_calls else "stop"
                chunk = _make_chunk(chat_id, model, created, {}, finish_reason=finish_reason)
                if event.response.usage:
                    chunk["usage"] = {
                        "prompt_tokens": event.response.usage.input_tokens,
                        "completion_tokens": event.response.usage.output_tokens,
                        "total_tokens": event.response.usage.total_tokens,
                    }
                yield {"data": json.dumps(chunk)}

            elif isinstance(event, ResponseFailedEvent):
                error = event.response.error
                yield {
                    "data": json.dumps(
                        {
                            "error": {
                                "message": error.message if error else "Backend response failed",
                                "type": error.type if error else "server_error",
                                "code": error.code if error else "server_error",
                            }
                        }
                    )
                }

    except Exception as e:
        error_payload = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "server_error",
            }
        }
        yield {"data": json.dumps(error_payload)}

    yield {"data": "[DONE]"}


# ---------------------------------------------------------------------------
# Outbound non-streaming
# ---------------------------------------------------------------------------


async def _non_stream_chat_response(chain, ctx, chat_id: str, model: str, created: int):
    final = await collect_terminal_response(chain, ctx)

    if final is None:
        return openai_error_response(
            status_code=502,
            message="No terminal response from backend",
            error_type="server_error",
            code="server_error",
        )
    if final.status is Status.FAILED:
        return openai_failed_response(final)

    text_parts: list[str] = []
    tool_calls_out: list[dict[str, Any]] = []

    for item in final.output:
        if isinstance(item, OutputMessage):
            for part in item.content:
                if isinstance(part, OutputTextContent):
                    text_parts.append(part.text)
        elif isinstance(item, FunctionCall):
            tool_calls_out.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )

    content = "\n".join(text_parts) if text_parts else None
    has_tool_calls = len(tool_calls_out) > 0
    finish_reason = "tool_calls" if has_tool_calls else "stop"

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls_out:
        message["tool_calls"] = tool_calls_out

    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(final.usage, Usage):
        usage = {
            "prompt_tokens": final.usage.input_tokens,
            "completion_tokens": final.usage.output_tokens,
            "total_tokens": final.usage.total_tokens,
        }

    return JSONResponse(
        content={
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ChatCompletionsRouter:
    """Router for the OpenAI Chat Completions-compatible interface."""

    def __init__(self, *, prefix: str = "/v1") -> None:
        self._prefix = prefix

    def build_router(self, mount: "SubspaceMount") -> APIRouter:
        router = APIRouter(prefix=self._prefix)
        subspace = mount.subspace
        interface_mw = list(mount.middlewares)
        deps_getter = mount.deps or no_deps
        context_class = mount.context_class

        deps_dependency = Depends(deps_getter)

        @router.post("/chat/completions")
        async def create_chat_completion(
            request: Request,
            body: ChatCompletionRequest,
            deps: Any = deps_dependency,
        ):
            internal_req = _to_internal_request(body)

            try:
                chain = subspace.build_chain(internal_req.model, interface_mw)
            except AgentNotFoundError:
                return openai_error_response(
                    status_code=404,
                    message=f"Model not found: {body.model}",
                    error_type="not_found",
                    code="model_not_found",
                )

            response_id = f"resp_{uuid.uuid4().hex[:24]}"
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            ctx = context_class(
                request=internal_req,
                response_id=response_id,
                response=ResponseResource(
                    id=response_id,
                    created_at=created,
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
                    _stream_chat_response(chain, ctx, chat_id, internal_req.model, created),
                    media_type="text/event-stream",
                )

            return await _non_stream_chat_response(chain, ctx, chat_id, internal_req.model, created)

        return router
