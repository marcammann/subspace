import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from conftest import function_call_backend_events, make_ctx, text_backend_events

from subspace.middleware.context import RequestContext
from subspace.middleware.function_call import FunctionCallMiddleware, server_tool
from subspace.models.common import Status
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseOutputItemDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, OutputMessage, ServerFunctionCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MultiBatchBackend:
    """Backend that yields a predetermined list of events."""

    def __init__(self, event_batches: list[list[StreamEvent]]) -> None:
        self._batches = list(event_batches)
        self._call_count = 0

    async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        batch = self._batches[min(self._call_count, len(self._batches) - 1)]
        self._call_count += 1
        for event in batch:
            yield event


# ---------------------------------------------------------------------------
# Tracking middleware — records all callback invocations
# ---------------------------------------------------------------------------


class TrackingMiddleware(FunctionCallMiddleware):
    """Test middleware that records every callback invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def prepare(self, ctx: RequestContext) -> None:
        self.calls.append(("prepare", ctx))

    async def on_function_call(self, ctx: RequestContext, call: FunctionCall) -> str | None:
        self.calls.append(("on_function_call", call))
        return None


async def _run_middleware(
    mw: FunctionCallMiddleware,
    backend: MultiBatchBackend,
    ctx: RequestContext | None = None,
) -> list[StreamEvent]:
    if ctx is None:
        ctx = make_ctx()

    from subspace.middleware.chain import MiddlewareChain

    chain = MiddlewareChain(middlewares=[mw], backend=backend)
    return [event async for event in chain.execute(ctx)]


# ---------------------------------------------------------------------------
# Tests: on_function_call
# ---------------------------------------------------------------------------


class TestOnFunctionCall:
    @pytest.mark.anyio
    async def test_on_function_call_called_for_owned_tool(self):
        """on_function_call is called when the middleware claims ownership via owns_function."""

        class OwnsWeather(FunctionCallMiddleware):
            def __init__(self):
                self.fc_calls: list[FunctionCall] = []

            def owns_function(self, name: str) -> bool:
                return name == "get_weather" or super().owns_function(name)

            async def on_function_call(self, ctx: RequestContext, call: FunctionCall) -> str | None:
                self.fc_calls.append(call)
                return None

        mw = OwnsWeather()
        backend = MultiBatchBackend([function_call_backend_events()])
        await _run_middleware(mw, backend)

        assert len(mw.fc_calls) == 1
        assert mw.fc_calls[0].name == "get_weather"

    @pytest.mark.anyio
    async def test_owned_function_returning_none_passes_buffered_events_through(self):
        class OwnsWeather(FunctionCallMiddleware):
            def owns_function(self, name: str) -> bool:
                return name == "get_weather" or super().owns_function(name)

            async def on_function_call(self, ctx: RequestContext, call: FunctionCall) -> str | None:
                return None

        mw = OwnsWeather()
        backend = MultiBatchBackend([function_call_backend_events()])
        events = await _run_middleware(mw, backend)

        assert any(
            isinstance(event, ResponseOutputItemDoneEvent)
            and isinstance(event.item, FunctionCall)
            for event in events
        )
        completed = [event for event in events if isinstance(event, ResponseCompletedEvent)]
        assert any(isinstance(item, FunctionCall) for item in completed[0].response.output)

    @pytest.mark.anyio
    async def test_unowned_function_passes_through_without_callback(self):
        """Functions not owned by the middleware pass through without on_function_call."""
        mw = TrackingMiddleware()
        backend = MultiBatchBackend([function_call_backend_events()])
        events = await _run_middleware(mw, backend)

        fc_calls = [c for c in mw.calls if c[0] == "on_function_call"]
        assert len(fc_calls) == 0

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        output = completed[0].response.output
        assert any(isinstance(item, FunctionCall) for item in output)

    @pytest.mark.anyio
    async def test_on_function_call_returning_result_triggers_roundtrip(self):
        class HandleWeather(FunctionCallMiddleware):
            def __init__(self):
                self.call_count = 0

            def owns_function(self, name: str) -> bool:
                return name == "get_weather" or super().owns_function(name)

            async def on_function_call(self, ctx: RequestContext, call: FunctionCall) -> str | None:
                if call.name == "get_weather":
                    self.call_count += 1
                    return '{"temp": 20}'
                return None

        mw = HandleWeather()
        text_events = text_backend_events("The weather is 20 degrees")
        fc_events = function_call_backend_events()
        backend = MultiBatchBackend([fc_events, text_events])
        events = await _run_middleware(mw, backend)

        assert mw.call_count == 1
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        output = completed[0].response.output
        assert any(isinstance(item, ServerFunctionCall) for item in output)
        assert any(isinstance(item, OutputMessage) for item in output)


# ---------------------------------------------------------------------------
# Tests: @server_tool decorator
# ---------------------------------------------------------------------------


class TestServerTool:
    @pytest.mark.anyio
    async def test_server_tool_injected_into_request(self):
        class WithTool(FunctionCallMiddleware):
            @server_tool(name="get_time", description="Get current time")
            async def get_time(self, ctx: RequestContext) -> str:
                return "2025-01-01T00:00:00Z"

        mw = WithTool()
        backend = MultiBatchBackend([text_backend_events()])
        ctx = make_ctx()
        await _run_middleware(mw, backend, ctx)

        tool_names = [t.name for t in (ctx.request.tools or [])]
        assert "get_time" in tool_names

    @pytest.mark.anyio
    async def test_server_tool_dispatched_with_kwargs(self):
        class WithTool(FunctionCallMiddleware):
            def __init__(self):
                self.received_kwargs: dict = {}

            @server_tool(name="greet", description="Greet someone")
            async def greet(self, ctx: RequestContext, name: str, loud: bool = False) -> str:
                self.received_kwargs = {"name": name, "loud": loud}
                return f"Hello, {name}!"

        mw = WithTool()
        fc_events = function_call_backend_events(
            name="greet",
            call_id="call_greet",
            arguments=json.dumps({"name": "Alice", "loud": True}),
        )
        text_events = text_backend_events("Hello, Alice!")
        backend = MultiBatchBackend([fc_events, text_events])
        events = await _run_middleware(mw, backend)

        assert mw.received_kwargs == {"name": "Alice", "loud": True}
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        output = completed[0].response.output
        server_calls = [item for item in output if isinstance(item, ServerFunctionCall)]
        assert len(server_calls) == 1
        assert server_calls[0].name == "greet"
        assert server_calls[0].output == "Hello, Alice!"

    @pytest.mark.anyio
    async def test_server_tool_emits_server_function_call(self):
        class WithTool(FunctionCallMiddleware):
            @server_tool(name="noop", description="Does nothing")
            async def noop(self, ctx: RequestContext) -> str:
                return "done"

        mw = WithTool()
        fc_events = function_call_backend_events(name="noop", call_id="call_noop", arguments="{}")
        text_events = text_backend_events("Ok")
        backend = MultiBatchBackend([fc_events, text_events])
        events = await _run_middleware(mw, backend)

        done_events = [
            e
            for e in events
            if isinstance(e, ResponseOutputItemDoneEvent)
            and isinstance(e.item, ServerFunctionCall)
        ]
        assert len(done_events) == 1
        assert done_events[0].item.name == "noop"


# ---------------------------------------------------------------------------
# Tests: unbundle ServerFunctionCall on inbound
# ---------------------------------------------------------------------------


class TestUnbundleServerCalls:
    @pytest.mark.anyio
    async def test_server_function_call_unbundled_for_backend(self):
        """ServerFunctionCall items owned by the middleware are unbundled into
        FunctionCall + FunctionCallOutput before the backend sees them."""

        class CapturingBackend:
            def __init__(self):
                self.seen_input: list = []

            async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
                self.seen_input = list(ctx.request.input)
                for event in text_backend_events():
                    yield event

        class WithTool(FunctionCallMiddleware):
            @server_tool(name="lookup", description="Lookup something")
            async def lookup(self, ctx: RequestContext, query: str) -> str:
                return "result"

        mw = WithTool()
        sfc = ServerFunctionCall(
            id="fc_1",
            name="lookup",
            call_id="call_1",
            arguments='{"query": "test"}',
            output="result",
        )
        ctx = make_ctx(input=[
            {"type": "message", "role": "user", "content": "look up test"},
            sfc,
        ])
        backend = CapturingBackend()
        from subspace.middleware.chain import MiddlewareChain

        chain = MiddlewareChain(middlewares=[mw], backend=backend)
        _ = [event async for event in chain.execute(ctx)]

        from subspace.models.items import FunctionCallOutput

        fc_items = [i for i in backend.seen_input if isinstance(i, FunctionCall)]
        fco_items = [i for i in backend.seen_input if isinstance(i, FunctionCallOutput)]
        assert len(fc_items) == 1
        assert fc_items[0].call_id == "call_1"
        assert len(fco_items) == 1
        assert fco_items[0].output == "result"


# ---------------------------------------------------------------------------
# Tests: callback ordering
# ---------------------------------------------------------------------------


class TestCallbackOrdering:
    @pytest.mark.anyio
    async def test_prepare_called_first(self):
        mw = TrackingMiddleware()
        backend = MultiBatchBackend([text_backend_events("hi")])
        await _run_middleware(mw, backend)

        names = [c[0] for c in mw.calls]
        assert names[0] == "prepare"

    @pytest.mark.anyio
    async def test_emits_completed_event(self):
        mw = TrackingMiddleware()
        backend = MultiBatchBackend([text_backend_events("hi")])
        events = await _run_middleware(mw, backend)

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].response.status == Status.COMPLETED
