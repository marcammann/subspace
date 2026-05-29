"""Backend that wraps any LangChain Runnable as a Subspace backend.

Accepts a static Runnable or a factory that receives RequestContext.

Usage:
    from subspace.contrib.backends.langchain import LangchainBackend

    def make_agent(ctx: RequestContext):
        return create_agent("anthropic:claude-sonnet-4-6", tools=[...])

    backend = LangchainBackend(make_agent)
"""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig

from subspace.backends.response_builder import ResponseBuilder
from subspace.middleware.context import RequestContext
from subspace.models.common import (
    InputTokensDetails,
    OutputTokensDetails,
    Usage,
)
from subspace.models.content import OutputTextContent
from subspace.models.events import StreamEvent
from subspace.models.items import (
    FunctionCall,
    FunctionCallOutput,
    InputMessage,
    OutputMessage,
)

logger = logging.getLogger("subspace.langchain")


class LangchainBackend:
    def __init__(self, runnable: Runnable | Callable[..., Runnable]) -> None:
        self._runnable = runnable

    async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        from subspace.contrib.backends.langchain.tools import make_interrupt_tools

        interrupt_tools = make_interrupt_tools(ctx.request.tools or [])
        _log_interrupt_tools(interrupt_tools)

        graph = self._resolve_runnable(ctx, interrupt_tools)
        config = _make_config(ctx)
        stream_input = self._build_stream_input(ctx)

        builder = ResponseBuilder(ctx.response)
        for event in builder.start():
            yield event

        try:
            translator = _StreamTranslator(ctx=ctx, builder=builder)
            async for part in graph.astream(
                stream_input,
                config=config,
                stream_mode=["messages", "updates"],
                version="v2",
            ):
                for event in translator.events_for_part(part):
                    yield event

            builder.set_usage(translator.usage)
            for event in builder.finish():
                yield event

        except Exception as exc:
            for event in builder.fail(exc):
                yield event

    def _build_stream_input(self, ctx: RequestContext) -> Any:
        """Build the input for graph.astream — either messages or a resume Command."""
        return _build_stream_input(ctx)

    def _resolve_runnable(self, ctx: RequestContext, interrupt_tools: list[Any]) -> Any:
        # A Runnable (or any graph-like object) is used directly;
        # a plain callable is treated as a factory.
        if isinstance(self._runnable, Runnable) or not callable(self._runnable):
            return self._runnable
        return self._runnable(ctx, interrupt_tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StreamTranslator:
    def __init__(self, *, ctx: RequestContext, builder: ResponseBuilder) -> None:
        self.ctx = ctx
        self.builder = builder
        self.usage = Usage()

    def events_for_part(self, part: dict[str, Any]) -> Iterator[StreamEvent]:
        part_type = part.get("type")
        if part_type == "messages":
            yield from self._events_for_message_part(part.get("data"))
        elif part_type == "updates":
            yield from self._events_for_update_part(part.get("data"))

    def _events_for_message_part(self, data: Any) -> Iterator[StreamEvent]:
        if not isinstance(data, (list, tuple)) or not data:
            return

        msg = data[0]
        if not isinstance(msg, AIMessageChunk):
            return

        if msg.usage_metadata:
            self.usage = _accumulate_usage(self.usage, msg.usage_metadata)

        if msg.tool_call_chunks:
            for chunk in msg.tool_call_chunks:
                yield from self.builder.tool_call_delta(
                    index=chunk.get("index", 0),
                    call_id=chunk.get("id"),
                    name=chunk.get("name"),
                    arguments=chunk.get("args"),
                )
            return

        for text in _extract_text_parts(msg):
            yield from self.builder.text_delta(text)

    def _events_for_update_part(self, data: Any) -> Iterator[StreamEvent]:
        if not isinstance(data, dict):
            return

        self._record_interrupts(data)
        for msg in _iter_update_messages(data):
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                output = _stringify_tool_output(msg.content)
                yield from self.builder.server_tool_output(msg.tool_call_id, output)

    def _record_interrupts(self, data: dict[str, Any]) -> None:
        interrupts = data.get("__interrupt__")
        if not interrupts:
            return

        interrupt_map: dict[str, str] = self.ctx.state.setdefault("_interrupt_map", {})
        call_ids = self.builder.call_ids()
        for intr in interrupts:
            idx = len(interrupt_map)
            if idx < len(call_ids):
                interrupt_map[call_ids[idx]] = intr.id
        logger.debug("interrupt_map: %s", interrupt_map)


def _log_interrupt_tools(interrupt_tools: list[Any]) -> None:
    for tool in interrupt_tools:
        logger.debug(
            "tool: %s — %s\n%s",
            tool.name,
            tool.description,
            tool.args_schema.model_json_schema(),
        )


def _make_config(ctx: RequestContext) -> RunnableConfig:
    thread_id = (
        (ctx.request.metadata or {}).get("thread_id")
        or ctx.metadata.get("thread_id")
        or ctx.response_id
    )
    return {"configurable": {"thread_id": thread_id}}


def _build_stream_input(ctx: RequestContext) -> Any:
    from langgraph.types import Command

    inp = ctx.request.input
    if not inp or not isinstance(inp[-1], FunctionCallOutput):
        return {"messages": _build_messages(ctx)}

    interrupt_map: dict[str, str] = ctx.state.pop("_interrupt_map", {})
    logger.debug("interrupt_map: %s", interrupt_map)
    if not interrupt_map:
        return {"messages": _build_messages(ctx)}

    outputs = _trailing_tool_outputs(inp)
    resume_map = {
        interrupt_map[out.call_id]: out.output for out in outputs if out.call_id in interrupt_map
    }
    return Command(resume=resume_map)


def _trailing_tool_outputs(items: list[Any]) -> list[FunctionCallOutput]:
    outputs: list[FunctionCallOutput] = []
    for item in reversed(items):
        if isinstance(item, FunctionCallOutput):
            outputs.append(item)
        else:
            break
    outputs.reverse()
    return outputs


def _iter_update_messages(data: dict[str, Any]) -> Iterator[Any]:
    for node_data in data.values():
        if isinstance(node_data, dict):
            yield from node_data.get("messages", [])


def _stringify_tool_output(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _extract_text_parts(chunk: Any) -> list[str]:
    content = chunk.content
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return parts
    return []


def _accumulate_usage(current: Usage, metadata: Any) -> Usage:
    input_details = None
    output_details = None

    raw_input = metadata.get("input_token_details")
    if raw_input:
        cached = raw_input.get("cache_read", 0) + raw_input.get("cache_creation", 0)
        prev = current.input_tokens_details
        input_details = InputTokensDetails(
            cached_tokens=(prev.cached_tokens if prev else 0) + cached
        )

    raw_output = metadata.get("output_token_details")
    if raw_output:
        prev = current.output_tokens_details
        output_details = OutputTokensDetails(
            reasoning_tokens=(prev.reasoning_tokens if prev else 0) + raw_output.get("reasoning", 0)
        )

    return Usage(
        input_tokens=current.input_tokens + metadata["input_tokens"],
        output_tokens=current.output_tokens + metadata["output_tokens"],
        total_tokens=current.total_tokens + metadata["total_tokens"],
        input_tokens_details=input_details or current.input_tokens_details,
        output_tokens_details=output_details or current.output_tokens_details,
    )


def _build_messages(ctx: RequestContext) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages: list[Any] = []

    if ctx.request.instructions:
        messages.append(SystemMessage(content=ctx.request.instructions))

    pending_tool_calls: list[FunctionCall] = []

    def _flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": fc.call_id, "name": fc.name, "args": _parse_tool_arguments(fc)}
                    for fc in pending_tool_calls
                ],
            )
        )
        pending_tool_calls.clear()

    for item in ctx.request.input:
        if isinstance(item, FunctionCall):
            pending_tool_calls.append(item)
            continue

        _flush_tool_calls()

        if isinstance(item, InputMessage):
            msg_class = {"user": HumanMessage, "system": SystemMessage}.get(item.role, AIMessage)
            messages.append(
                msg_class(
                    content=item.content if isinstance(item.content, str) else str(item.content)
                )
            )
        elif isinstance(item, OutputMessage):
            joined = "".join(
                part.text for part in item.content if isinstance(part, OutputTextContent)
            )
            messages.append(AIMessage(content=joined))
        elif isinstance(item, FunctionCallOutput):
            messages.append(ToolMessage(content=item.output, tool_call_id=item.call_id))

    _flush_tool_calls()

    return messages


def _parse_tool_arguments(call: FunctionCall) -> dict[str, Any]:
    try:
        value = json.loads(call.arguments) if call.arguments else {}
    except json.JSONDecodeError:
        logger.warning("Invalid JSON arguments for tool call %s", call.call_id)
        return {}
    return value if isinstance(value, dict) else {}
