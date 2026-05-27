import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from subspace.middleware.base import NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.common import ResponseError, Status, Usage
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, OutputMessage
from subspace.models.tools import FunctionTool

if TYPE_CHECKING:
    from subspace.core import Subspace

TOOL_NAME = "delegate_to"

TOOL_DEFINITION = FunctionTool(
    name=TOOL_NAME,
    description="Hand off the conversation to another model that is better suited to respond.",
    parameters={
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "The name of the model to delegate to.",
            },
            "reason": {
                "type": "string",
                "description": "Why this model is better suited to handle the request.",
            },
        },
        "required": ["model"],
        "additionalProperties": False,
    },
)


class DelegateMiddleware:
    """Middleware that enables model-to-model delegation.

    Injects a `delegate_to` tool that allows a model to hand off the conversation
    to another model registered in the Subspace registry. The target model sees
    the full conversation history so it can seamlessly take over.
    """

    def __init__(
        self,
        subspace: "Subspace",
        available_models: list[str] | None = None,
    ) -> None:
        self._subspace = subspace
        self._available_models = available_models

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        tool = self._build_tool()
        existing_tools = ctx.request.tools or []
        ctx.request = ctx.request.model_copy(update={"tools": existing_tools + [tool]})

        delegate_call: FunctionCall | None = None
        delegate_item_ids: set[str] = set()
        all_output: list[OutputMessage | FunctionCall] = []
        total_usage = Usage()
        seq = 0

        async for event in call_next(ctx):
            seq = max(seq, event.sequence_number) + 1
            if (
                isinstance(event, ResponseOutputItemAddedEvent)
                and isinstance(event.item, FunctionCall)
                and event.item.name == TOOL_NAME
            ):
                delegate_item_ids.add(event.item.id)
                continue

            if _is_delegate_event(event, delegate_item_ids):
                if isinstance(event, ResponseOutputItemDoneEvent):
                    delegate_call = event.item
                continue

            if isinstance(event, ResponseCompletedEvent):
                if event.response.usage:
                    u = event.response.usage
                    total_usage = Usage(
                        input_tokens=total_usage.input_tokens + u.input_tokens,
                        output_tokens=total_usage.output_tokens + u.output_tokens,
                        total_tokens=total_usage.total_tokens + u.total_tokens,
                    )
                continue

            if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
                event.item, (OutputMessage, FunctionCall)
            ):
                all_output.append(event.item)

            yield event

        if delegate_call is not None:
            target_model = _parse_target(delegate_call)

            if target_model is None or self._subspace.resolve(target_model) is None:
                yield _error_completed(
                    ctx,
                    seq,
                    all_output,
                    total_usage,
                    f"Cannot delegate: model '{target_model}' not found",
                )
                return

            delegate_input = ctx.request.input + list(all_output)

            delegate_request = ctx.request.model_copy(
                update={
                    "model": target_model,
                    "input": delegate_input,
                    "tools": None,
                    "instructions": None,
                }
            )
            delegate_ctx = RequestContext(
                request=delegate_request,
                response_id=ctx.response_id,
                response=ctx.response.model_copy(
                    update={"model": target_model, "tools": None, "instructions": None}
                ),
                app=ctx.app,
                metadata={**ctx.metadata, "delegated_from": ctx.request.model},
            )

            chain = self._subspace.build_chain(target_model)
            offset = len(all_output)

            async for event in chain.execute(delegate_ctx):
                seq = max(seq, event.sequence_number) + 1

                if isinstance(event, (ResponseCreatedEvent, ResponseInProgressEvent)):
                    continue

                if isinstance(event, ResponseCompletedEvent):
                    if event.response.usage:
                        u = event.response.usage
                        total_usage = Usage(
                            input_tokens=total_usage.input_tokens + u.input_tokens,
                            output_tokens=total_usage.output_tokens + u.output_tokens,
                            total_tokens=total_usage.total_tokens + u.total_tokens,
                        )
                    continue

                if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
                    event.item, (OutputMessage, FunctionCall)
                ):
                    all_output.append(event.item)

                event = _offset_output_index(event, offset)
                yield event

        final_response = ctx.response.model_copy(
            update={
                "status": Status.COMPLETED,
                "output": all_output,
                "usage": total_usage,
            }
        )
        yield ResponseCompletedEvent(sequence_number=seq, response=final_response)

    def _build_tool(self) -> FunctionTool:
        models = self._available_models or self._subspace.list_models()
        tool = TOOL_DEFINITION.model_copy()
        params = dict(tool.parameters) if tool.parameters else {}
        props = dict(params.get("properties", {}))
        props["model"] = {**props["model"], "enum": models}
        params["properties"] = props
        tool.parameters = params
        return tool


def _is_delegate_event(event: StreamEvent, delegate_ids: set[str]) -> bool:
    if (
        isinstance(event, (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent))
        and isinstance(event.item, (FunctionCall, OutputMessage))
        and event.item.id in delegate_ids
    ):
        return True
    return (
        isinstance(event, (ResponseFunctionCallArgsDeltaEvent, ResponseFunctionCallArgsDoneEvent))
        and event.item_id in delegate_ids
    )


def _parse_target(call: FunctionCall) -> str | None:
    try:
        args = json.loads(call.arguments)
        return args.get("model")
    except (json.JSONDecodeError, AttributeError):
        return None


def _offset_output_index(event: StreamEvent, offset: int) -> StreamEvent:
    if offset == 0:
        return event

    if isinstance(
        event,
        (
            ResponseOutputItemAddedEvent,
            ResponseOutputItemDoneEvent,
            ResponseContentPartAddedEvent,
            ResponseContentPartDoneEvent,
            ResponseOutputTextDeltaEvent,
            ResponseOutputTextDoneEvent,
            ResponseFunctionCallArgsDeltaEvent,
            ResponseFunctionCallArgsDoneEvent,
        ),
    ):
        return event.model_copy(update={"output_index": event.output_index + offset})

    return event


def _error_completed(
    ctx: RequestContext,
    seq: int,
    output: list,
    usage: Usage,
    message: str,
) -> ResponseCompletedEvent:
    return ResponseCompletedEvent(
        sequence_number=seq,
        response=ctx.response.model_copy(
            update={
                "status": Status.FAILED,
                "output": output,
                "usage": usage,
                "error": ResponseError(
                    message=message, type="invalid_request", code="model_not_found"
                ),
            }
        ),
    )
