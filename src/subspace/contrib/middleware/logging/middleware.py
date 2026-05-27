import logging
import time
from collections.abc import AsyncIterator

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.events import (
    ErrorEvent,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, OutputMessage, ServerFunctionCall

logger = logging.getLogger("subspace")


def _truncate(s: str, n: int = 60) -> str:
    return s[:n] + "..." if len(s) > n else s


def _format_event(event: StreamEvent) -> str:
    match event:
        case ResponseOutputItemAddedEvent(item=item):
            if isinstance(item, OutputMessage):
                return f"+ message {item.id}"
            if isinstance(item, FunctionCall):
                return f"+ function_call {item.name} {item.call_id}"
            if isinstance(item, ServerFunctionCall):
                return f"+ server_tool_call {item.name} {item.call_id}"
            return f"+ {item.type}"

        case ResponseOutputItemDoneEvent(item=item):
            if isinstance(item, OutputMessage):
                text = "".join(p.text for p in item.content if hasattr(p, "text"))
                return f"  message done: {_truncate(text)}"
            if isinstance(item, FunctionCall):
                return f"  function_call done: {item.name}({_truncate(item.arguments, 40)})"
            if isinstance(item, ServerFunctionCall):
                return (
                    f"  server_tool done: {item.name}({_truncate(item.arguments, 40)})"
                    f" -> {_truncate(item.output, 40)}"
                )
            return f"  {item.type} done"

        case ResponseOutputTextDeltaEvent(delta=delta):
            return f"  text delta: {_truncate(delta, 40)}"

        case ResponseOutputTextDoneEvent(text=text):
            return f"  text done ({len(text)} chars)"

        case ResponseContentPartAddedEvent():
            return "  content_part added"

        case ResponseContentPartDoneEvent():
            return "  content_part done"

        case ResponseFunctionCallArgsDeltaEvent(delta=delta):
            return f"  args delta: {_truncate(delta, 40)}"

        case ResponseFunctionCallArgsDoneEvent(arguments=arguments):
            return f"  args done: {_truncate(arguments, 50)}"

        case ResponseCompletedEvent(response=resp):
            n = len(resp.output) if resp.output else 0
            usage = ""
            if resp.usage:
                usage = f" ({resp.usage.input_tokens}in/{resp.usage.output_tokens}out)"
            return f"completed: {n} items{usage}"

        case ResponseFailedEvent(response=resp):
            msg = resp.error.message if resp.error else "unknown"
            return f"failed: {msg}"

        case ErrorEvent(message=msg):
            return f"error: {msg}"

        case _:
            return event.type


class LoggingMiddleware(Middleware):
    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        model = ctx.request.model
        rid = ctx.response_id[:16]
        start = time.monotonic()

        logger.info("__call__ >>> %s %s", model, rid)

        async for event in call_next(ctx):
            logger.info("  [%s] %s", rid, _format_event(event))
            yield event

        elapsed = time.monotonic() - start
        logger.info("__call__ <<< %s %s %.2fs", model, rid, elapsed)
