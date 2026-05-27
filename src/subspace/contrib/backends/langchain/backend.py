"""Backend that wraps any LangChain Runnable as a Subspace backend.

Accepts a static Runnable or a factory that receives RequestContext.

Usage:
    from subspace.contrib.backends.langchain import LangchainBackend

    def make_agent(ctx: RequestContext):
        return create_agent("anthropic:claude-sonnet-4-6", tools=[...])

    backend = LangchainBackend(make_agent)
"""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import AIMessageChunk
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
        graph = self._resolve_runnable(ctx, interrupt_tools)
        config = _make_config(ctx)
        stream_input = self._build_stream_input(ctx)

        builder = ResponseBuilder(ctx.response)
        for event in builder.start():
            yield event

        usage = Usage()

        try:
            async for part in graph.astream(
                stream_input,
                config=config,
                stream_mode=["messages", "updates"],
                version="v2",
            ):
                if part["type"] == "messages":
                    msg, _ = part["data"]
                    if not isinstance(msg, AIMessageChunk):
                        continue

                    if msg.usage_metadata:
                        usage = _accumulate_usage(usage, msg.usage_metadata)

                    if msg.tool_call_chunks:
                        for tc in msg.tool_call_chunks:
                            for event in builder.tool_call_delta(
                                index=tc.get("index", 0),
                                call_id=tc.get("id"),
                                name=tc.get("name"),
                                arguments=tc.get("args"),
                            ):
                                yield event
                    else:
                        for text in _extract_text_parts(msg):
                            for event in builder.text_delta(text):
                                yield event

                elif part["type"] == "updates" and "__interrupt__" in part["data"]:
                    interrupt_map: dict[str, str] = ctx.state.setdefault(
                        "_interrupt_map", {}
                    )
                    for intr in part["data"]["__interrupt__"]:
                        call_ids = builder.call_ids()
                        idx = len(interrupt_map)
                        if idx < len(call_ids):
                            interrupt_map[call_ids[idx]] = intr.id
                    logger.debug("interrupt_map: %s", interrupt_map)

            builder.set_usage(usage)
            for event in builder.finish():
                yield event

        except Exception as exc:
            for event in builder.fail(exc):
                yield event

    def _build_stream_input(self, ctx: RequestContext) -> Any:
        """Build the input for graph.astream — either messages or a resume Command."""
        from langgraph.types import Command

        inp = ctx.request.input
        if not inp or not isinstance(inp[-1], FunctionCallOutput):
            return {"messages": _build_messages(ctx)}

        outputs: list[FunctionCallOutput] = []
        for item in reversed(inp):
            if isinstance(item, FunctionCallOutput):
                outputs.append(item)
            else:
                break
        outputs.reverse()

        interrupt_map: dict[str, str] = ctx.state.pop("_interrupt_map", {})
        resume_map = {
            interrupt_map[out.call_id]: out.output
            for out in outputs
            if out.call_id in interrupt_map
        }
        return Command(resume=resume_map)

    def _resolve_runnable(self, ctx: RequestContext, interrupt_tools: list[Any]) -> Any:
        if isinstance(self._runnable, Runnable):
            return self._runnable
        return self._runnable(ctx, interrupt_tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(ctx: RequestContext) -> RunnableConfig:
    thread_id = (
        (ctx.request.metadata or {}).get("thread_id")
        or ctx.metadata.get("thread_id")
        or ctx.response_id
    )
    return {"configurable": {"thread_id": thread_id}}


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
            reasoning_tokens=(prev.reasoning_tokens if prev else 0)
            + raw_output.get("reasoning", 0)
        )

    return Usage(
        input_tokens=current.input_tokens + metadata["input_tokens"],
        output_tokens=current.output_tokens + metadata["output_tokens"],
        total_tokens=current.total_tokens + metadata["total_tokens"],
        input_tokens_details=input_details or current.input_tokens_details,
        output_tokens_details=output_details or current.output_tokens_details,
    )


def _build_messages(ctx: RequestContext) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    messages: list[Any] = []

    if ctx.request.instructions:
        messages.append(SystemMessage(content=ctx.request.instructions))

    for item in ctx.request.input:
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

    return messages
