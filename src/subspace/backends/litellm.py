import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import litellm

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper

from subspace.backends.response_builder import ResponseBuilder
from subspace.middleware.context import RequestContext
from subspace.models.agent import AgentCapabilities
from subspace.models.common import Usage
from subspace.models.content import (
    InputImageContent,
    InputTextContent,
    OutputTextContent,
)
from subspace.models.events import StreamEvent
from subspace.models.items import (
    FunctionCall,
    FunctionCallOutput,
    InputMessage,
    OutputMessage,
)

logger = logging.getLogger("subspace.litellm")


class LitellmBackend:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._extra = extra or {}

    @property
    def capabilities(self) -> AgentCapabilities:
        """Capabilities provided by LiteLLM-compatible chat providers."""
        return AgentCapabilities(
            streaming=True,
            text_input=True,
            image_input=True,
            function_tools=True,
        )

    async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        builder = ResponseBuilder(ctx.response)
        for event in builder.start():
            yield event

        messages = _build_messages(ctx)
        kwargs = self._build_kwargs(ctx, messages)

        try:
            stream = cast("CustomStreamWrapper", await litellm.acompletion(**kwargs))

            async for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    builder.set_usage(
                        Usage(
                            input_tokens=chunk_usage.prompt_tokens or 0,
                            output_tokens=chunk_usage.completion_tokens or 0,
                            total_tokens=(chunk_usage.prompt_tokens or 0)
                            + (chunk_usage.completion_tokens or 0),
                        )
                    )

                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta

                if delta.content:
                    for event in builder.text_delta(delta.content):
                        yield event

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        for event in builder.tool_call_delta(
                            index=tc.index if tc.index is not None else 0,
                            call_id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ):
                            yield event

            for event in builder.finish():
                yield event

        except Exception as exc:
            for event in builder.fail(exc):
                yield event

    def _build_kwargs(self, ctx: RequestContext, messages: list[dict[str, Any]]) -> dict[str, Any]:
        req = ctx.request
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.max_output_tokens is not None:
            kwargs["max_tokens"] = req.max_output_tokens

        if req.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.parameters or {"type": "object"},
                    },
                }
                for t in req.tools
            ]

        if req.tool_choice is not None:
            tc = req.tool_choice
            if isinstance(tc, str):
                kwargs["tool_choice"] = tc
            else:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.name},
                }

        if req.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = req.parallel_tool_calls

        kwargs.update(self._extra)

        return kwargs


def _build_messages(ctx: RequestContext) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if ctx.request.instructions:
        messages.append({"role": "system", "content": ctx.request.instructions})

    pending_tool_calls: list[FunctionCall] = []

    def _flush_tool_calls():
        if not pending_tool_calls:
            return
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": fc.call_id,
                        "type": "function",
                        "function": {"name": fc.name, "arguments": fc.arguments},
                    }
                    for fc in pending_tool_calls
                ],
            }
        )
        pending_tool_calls.clear()

    for item in ctx.request.input:
        if isinstance(item, FunctionCall):
            pending_tool_calls.append(item)
            continue

        _flush_tool_calls()

        if isinstance(item, InputMessage):
            if isinstance(item.content, str):
                messages.append({"role": item.role, "content": item.content})
            else:
                content_parts = []
                for p in item.content:
                    if isinstance(p, InputTextContent):
                        content_parts.append({"type": "text", "text": p.text})
                    elif isinstance(p, InputImageContent):
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": p.image_url}}
                        )
                messages.append({"role": item.role, "content": content_parts})

        elif isinstance(item, OutputMessage):
            text = ""
            for part in item.content:
                if isinstance(part, OutputTextContent):
                    text += part.text
            messages.append({"role": "assistant", "content": text})

        elif isinstance(item, FunctionCallOutput):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output,
                }
            )

    _flush_tool_calls()

    return messages
