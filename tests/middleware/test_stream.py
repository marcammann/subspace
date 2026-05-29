from typing import Any

import pytest
from conftest import StaticBackend, error_backend_events, make_ctx, text_backend_events

from subspace.middleware.chain import MiddlewareChain
from subspace.middleware.context import RequestContext
from subspace.middleware.stream import StreamMiddleware
from subspace.models.common import Status
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ErrorEvent,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import OutputMessage
from subspace.models.response import ResponseResource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run(mw: StreamMiddleware, backend_events: list[StreamEvent]) -> list[StreamEvent]:
    ctx = make_ctx()
    chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(backend_events))
    return [event async for event in chain.execute(ctx)]


# ---------------------------------------------------------------------------
# Tracking middleware
# ---------------------------------------------------------------------------


class TrackingMiddleware(StreamMiddleware):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def on_request(self, ctx: RequestContext) -> None:
        self.calls.append(("on_request", ctx))

    async def on_event(self, ctx: RequestContext, event: StreamEvent) -> StreamEvent | None:
        self.calls.append(("on_event", event))
        return event

    async def on_output_item_done(self, ctx, item):
        self.calls.append(("on_output_item_done", item))
        return item

    async def on_response_completed(self, ctx: RequestContext, response: ResponseResource):
        self.calls.append(("on_response_completed", response))
        return response


# ---------------------------------------------------------------------------
# Tests: basic lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.anyio
    async def test_callback_order(self):
        mw = TrackingMiddleware()
        await _run(mw, text_backend_events("hi"))

        names = [c[0] for c in mw.calls]
        assert names[0] == "on_request"
        assert "on_output_item_done" in names
        assert names[-1] == "on_response_completed"

    @pytest.mark.anyio
    async def test_emits_own_completed_event(self):
        mw = StreamMiddleware()
        events = await _run(mw, text_backend_events())

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].response.status == Status.COMPLETED

    @pytest.mark.anyio
    async def test_usage_accumulated(self):
        mw = StreamMiddleware()
        events = await _run(mw, text_backend_events())

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        assert completed.response.usage.input_tokens == 2
        assert completed.response.usage.output_tokens == 2

    @pytest.mark.anyio
    async def test_backend_completed_event_swallowed(self):
        """The backend's CompletedEvent is consumed; only the middleware's is emitted."""
        mw = TrackingMiddleware()
        await _run(mw, text_backend_events())

        on_event_types = {type(c[1]) for c in mw.calls if c[0] == "on_event"}
        assert ResponseCompletedEvent not in on_event_types


# ---------------------------------------------------------------------------
# Tests: on_event suppression
# ---------------------------------------------------------------------------


class TestOnEventSuppression:
    @pytest.mark.anyio
    async def test_suppress_text_deltas(self):
        class NoDelta(StreamMiddleware):
            async def on_event(self, ctx, event):
                if isinstance(event, ResponseOutputTextDeltaEvent):
                    return None
                return event

        events = await _run(NoDelta(), text_backend_events("Hello world"))
        assert not any(isinstance(e, ResponseOutputTextDeltaEvent) for e in events)
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)


# ---------------------------------------------------------------------------
# Tests: on_output_item_done
# ---------------------------------------------------------------------------


class TestOnOutputItem:
    @pytest.mark.anyio
    async def test_called_with_completed_item(self):
        mw = TrackingMiddleware()
        await _run(mw, text_backend_events("hi"))

        items = [c[1] for c in mw.calls if c[0] == "on_output_item_done"]
        assert len(items) == 1
        assert isinstance(items[0], OutputMessage)
        assert items[0].status == Status.COMPLETED

    @pytest.mark.anyio
    async def test_drop_item(self):
        class DropAll(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                return None

        events = await _run(DropAll(), text_backend_events())

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        assert completed.response.output == []
        assert not any(isinstance(e, ResponseOutputItemDoneEvent) for e in events)

    @pytest.mark.anyio
    async def test_transform_item(self):
        class Redact(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                if isinstance(item, OutputMessage):
                    return item.model_copy(
                        update={"content": [OutputTextContent(text="[REDACTED]")]}
                    )
                return item

        events = await _run(Redact(), text_backend_events("secret stuff"))

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        msg = completed.response.output[0]
        assert isinstance(msg, OutputMessage)
        assert msg.content[0].text == "[REDACTED]"

        done_events = [e for e in events if isinstance(e, ResponseOutputItemDoneEvent)]
        assert done_events[0].item.content[0].text == "[REDACTED]"

    @pytest.mark.anyio
    async def test_buffer_then_decide(self):
        """Suppress streaming events, then emit the item via on_output_item_done."""

        class BufferMiddleware(StreamMiddleware):
            async def on_event(self, ctx, event):
                if isinstance(
                    event,
                    (
                        ResponseOutputItemAddedEvent,
                        ResponseContentPartAddedEvent,
                        ResponseContentPartDoneEvent,
                        ResponseOutputTextDeltaEvent,
                        ResponseOutputTextDoneEvent,
                    ),
                ):
                    return None
                return event

            async def on_output_item_done(self, ctx, item):
                return item

        events = await _run(BufferMiddleware(), text_backend_events("hello"))

        assert not any(isinstance(e, ResponseOutputTextDeltaEvent) for e in events)
        assert not any(isinstance(e, ResponseOutputItemAddedEvent) for e in events)
        done_events = [e for e in events if isinstance(e, ResponseOutputItemDoneEvent)]
        assert len(done_events) == 1

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        assert len(completed.response.output) == 1


# ---------------------------------------------------------------------------
# Tests: error events flow through on_event
# ---------------------------------------------------------------------------


class TestErrorEvents:
    @pytest.mark.anyio
    async def test_failed_event_flows_through_on_event(self):
        mw = TrackingMiddleware()
        await _run(mw, error_backend_events())

        event_types = [type(c[1]) for c in mw.calls if c[0] == "on_event"]
        assert ResponseFailedEvent in event_types

    @pytest.mark.anyio
    async def test_error_event_flows_through_on_event(self):
        error_event = ErrorEvent(sequence_number=0, code="rate_limit", message="slow down")

        mw = TrackingMiddleware()
        await _run(mw, [error_event])

        event_types = [type(c[1]) for c in mw.calls if c[0] == "on_event"]
        assert ErrorEvent in event_types

    @pytest.mark.anyio
    async def test_error_can_be_suppressed_via_on_event(self):
        class SuppressErrors(StreamMiddleware):
            async def on_event(self, ctx, event):
                if isinstance(event, ResponseFailedEvent):
                    return None
                return event

        events = await _run(SuppressErrors(), error_backend_events())
        assert not any(isinstance(e, ResponseFailedEvent) for e in events)


# ---------------------------------------------------------------------------
# Tests: on_request
# ---------------------------------------------------------------------------


class TestOnRequest:
    @pytest.mark.anyio
    async def test_on_request_can_modify_ctx(self):
        class InjectInstructions(StreamMiddleware):
            async def on_request(self, ctx):
                ctx.request = ctx.request.model_copy(update={"instructions": "be helpful"})

        mw = InjectInstructions()
        ctx = make_ctx()
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(text_backend_events()))
        _ = [event async for event in chain.execute(ctx)]

        assert ctx.request.instructions == "be helpful"


# ---------------------------------------------------------------------------
# Tests: passthrough (no overrides)
# ---------------------------------------------------------------------------


class TestPassthrough:
    @pytest.mark.anyio
    async def test_default_streams_everything_through(self):
        mw = StreamMiddleware()
        backend_events = text_backend_events("Hello world")
        events = await _run(mw, backend_events)

        assert any(isinstance(e, ResponseCreatedEvent) for e in events)
        assert any(isinstance(e, ResponseOutputTextDeltaEvent) for e in events)
        assert any(isinstance(e, ResponseOutputItemDoneEvent) for e in events)
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    @pytest.mark.anyio
    async def test_output_items_in_completed_response(self):
        mw = StreamMiddleware()
        events = await _run(mw, text_backend_events("hi"))

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        assert len(completed.response.output) == 1
        assert isinstance(completed.response.output[0], OutputMessage)


# ---------------------------------------------------------------------------
# Tests: nested chain execution (context isolation)
# ---------------------------------------------------------------------------


class TestContextIsolation:
    @pytest.mark.anyio
    async def test_nested_execute_does_not_clobber_outer_state(self):
        """Simulates delegation: an inner chain.execute creates a new RequestContext
        without wiping the outer context's state."""

        class DelegatingMiddleware(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                ctx.state["outer_key"] = "outer_value"

                inner_ctx = make_ctx("inner")
                inner_ctx.state["inner_key"] = "inner_value"
                inner_chain = MiddlewareChain(
                    middlewares=[StreamMiddleware()],
                    backend=StaticBackend(text_backend_events("inner")),
                )
                async for _ in inner_chain.execute(inner_ctx):
                    pass

                assert ctx.state.get("outer_key") == "outer_value"
                assert "inner_key" not in ctx.state
                return item

        mw = DelegatingMiddleware()
        ctx = make_ctx()
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(text_backend_events("hi")))
        events = [event async for event in chain.execute(ctx)]

        assert any(isinstance(e, ResponseCompletedEvent) for e in events)
        assert ctx.state.get("outer_key") == "outer_value"

    @pytest.mark.anyio
    async def test_request_state_contextvar_restored_after_nested_execute(self):
        """The ContextVar points back to the outer ctx's state after the inner chain finishes."""
        from subspace.middleware.context import request_state

        contextvar_after_inner: dict[str, Any] = {}

        class InnerExecuteMiddleware(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                ctx.state["marker"] = "outer"

                inner_ctx = make_ctx("inner")
                inner_chain = MiddlewareChain(
                    middlewares=[StreamMiddleware()],
                    backend=StaticBackend(text_backend_events("inner")),
                )
                async for _ in inner_chain.execute(inner_ctx):
                    pass

                contextvar_after_inner.update(request_state.get())
                return item

        mw = InnerExecuteMiddleware()
        ctx = make_ctx()
        chain = MiddlewareChain(
            middlewares=[mw],
            backend=StaticBackend(text_backend_events("hi")),
        )
        async for _ in chain.execute(ctx):
            pass

        assert contextvar_after_inner.get("marker") == "outer"
