import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

from subspace.middleware.chain import MiddlewareChain
from subspace.middleware.context import RequestContext
from subspace.middleware.stream import StreamMiddleware
from subspace.models.common import ResponseError, Status, Usage
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ErrorEvent,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    StreamEvent,
)
from subspace.models.items import OutputMessage
from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(input: str | list = "hello") -> RequestContext:
    request = CreateResponseRequest(model="test", input=input)
    response = ResponseResource(
        id="resp_test",
        created_at=int(time.time()),
        status=Status.IN_PROGRESS,
        model="test",
    )
    return RequestContext(
        request=request,
        response_id="resp_test",
        response=response,
    )


def _text_backend_events(text: str = "Hello world") -> list[StreamEvent]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    seq = 0
    response = ResponseResource(
        id="resp_test", created_at=int(time.time()), status=Status.IN_PROGRESS, model="test"
    )
    events: list[StreamEvent] = []

    events.append(ResponseCreatedEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(ResponseInProgressEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(
        ResponseOutputItemAddedEvent(
            sequence_number=seq,
            output_index=0,
            item=OutputMessage(id=msg_id, content=[], status=Status.IN_PROGRESS),
        )
    )
    seq += 1
    events.append(
        ResponseContentPartAddedEvent(
            sequence_number=seq, item_id=msg_id, output_index=0, content_index=0,
            part=OutputTextContent(text=""),
        )
    )
    seq += 1
    for word in text.split():
        events.append(
            ResponseOutputTextDeltaEvent(
                sequence_number=seq, item_id=msg_id, output_index=0,
                content_index=0, delta=word + " ",
            )
        )
        seq += 1
    events.append(
        ResponseOutputTextDoneEvent(
            sequence_number=seq, item_id=msg_id, output_index=0,
            content_index=0, text=text,
        )
    )
    seq += 1
    events.append(
        ResponseContentPartDoneEvent(
            sequence_number=seq, item_id=msg_id, output_index=0,
            content_index=0, part=OutputTextContent(text=text),
        )
    )
    seq += 1
    done_msg = OutputMessage(
        id=msg_id, content=[OutputTextContent(text=text)], status=Status.COMPLETED,
    )
    events.append(ResponseOutputItemDoneEvent(sequence_number=seq, output_index=0, item=done_msg))
    seq += 1
    completed_response = response.model_copy(
        update={
            "status": Status.COMPLETED,
            "output": [done_msg],
            "usage": Usage(input_tokens=2, output_tokens=2, total_tokens=4),
        }
    )
    events.append(ResponseCompletedEvent(sequence_number=seq, response=completed_response))
    return events


def _error_backend_events() -> list[StreamEvent]:
    seq = 0
    response = ResponseResource(
        id="resp_test", created_at=int(time.time()), status=Status.IN_PROGRESS, model="test"
    )
    events: list[StreamEvent] = []
    events.append(ResponseCreatedEvent(sequence_number=seq, response=response))
    seq += 1
    events.append(ResponseInProgressEvent(sequence_number=seq, response=response))
    seq += 1
    failed_response = response.model_copy(
        update={
            "status": Status.FAILED,
            "error": ResponseError(message="boom", type="server_error", code="server_error"),
        }
    )
    events.append(ResponseFailedEvent(sequence_number=seq, response=failed_response))
    return events


class _StaticBackend:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


async def _run(mw: StreamMiddleware, backend_events: list[StreamEvent]) -> list[StreamEvent]:
    ctx = _make_ctx()
    chain = MiddlewareChain(middlewares=[mw], backend=_StaticBackend(backend_events))
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
    @pytest.mark.asyncio
    async def test_callback_order(self):
        mw = TrackingMiddleware()
        await _run(mw, _text_backend_events("hi"))

        names = [c[0] for c in mw.calls]
        assert names[0] == "on_request"
        assert "on_output_item_done" in names
        assert names[-1] == "on_response_completed"

    @pytest.mark.asyncio
    async def test_emits_own_completed_event(self):
        mw = StreamMiddleware()
        events = await _run(mw, _text_backend_events())

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].response.status == Status.COMPLETED

    @pytest.mark.asyncio
    async def test_usage_accumulated(self):
        mw = StreamMiddleware()
        events = await _run(mw, _text_backend_events())

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        assert completed.response.usage.input_tokens == 2
        assert completed.response.usage.output_tokens == 2

    @pytest.mark.asyncio
    async def test_backend_completed_event_swallowed(self):
        """The backend's CompletedEvent is consumed; only the middleware's is emitted."""
        mw = TrackingMiddleware()
        await _run(mw, _text_backend_events())

        on_event_types = {type(c[1]) for c in mw.calls if c[0] == "on_event"}
        assert ResponseCompletedEvent not in on_event_types


# ---------------------------------------------------------------------------
# Tests: on_event suppression
# ---------------------------------------------------------------------------


class TestOnEventSuppression:
    @pytest.mark.asyncio
    async def test_suppress_text_deltas(self):
        class NoDelta(StreamMiddleware):
            async def on_event(self, ctx, event):
                if isinstance(event, ResponseOutputTextDeltaEvent):
                    return None
                return event

        events = await _run(NoDelta(), _text_backend_events("Hello world"))
        assert not any(isinstance(e, ResponseOutputTextDeltaEvent) for e in events)
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)


# ---------------------------------------------------------------------------
# Tests: on_output_item_done
# ---------------------------------------------------------------------------


class TestOnOutputItem:
    @pytest.mark.asyncio
    async def test_called_with_completed_item(self):
        mw = TrackingMiddleware()
        await _run(mw, _text_backend_events("hi"))

        items = [c[1] for c in mw.calls if c[0] == "on_output_item_done"]
        assert len(items) == 1
        assert isinstance(items[0], OutputMessage)
        assert items[0].status == Status.COMPLETED

    @pytest.mark.asyncio
    async def test_drop_item(self):
        class DropAll(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                return None

        events = await _run(DropAll(), _text_backend_events())

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        assert completed.response.output == []
        assert not any(isinstance(e, ResponseOutputItemDoneEvent) for e in events)

    @pytest.mark.asyncio
    async def test_transform_item(self):
        class Redact(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                if isinstance(item, OutputMessage):
                    return item.model_copy(
                        update={"content": [OutputTextContent(text="[REDACTED]")]}
                    )
                return item

        events = await _run(Redact(), _text_backend_events("secret stuff"))

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        msg = completed.response.output[0]
        assert isinstance(msg, OutputMessage)
        assert msg.content[0].text == "[REDACTED]"

        done_events = [e for e in events if isinstance(e, ResponseOutputItemDoneEvent)]
        assert done_events[0].item.content[0].text == "[REDACTED]"

    @pytest.mark.asyncio
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

        events = await _run(BufferMiddleware(), _text_backend_events("hello"))

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
    @pytest.mark.asyncio
    async def test_failed_event_flows_through_on_event(self):
        mw = TrackingMiddleware()
        await _run(mw, _error_backend_events())

        event_types = [type(c[1]) for c in mw.calls if c[0] == "on_event"]
        assert ResponseFailedEvent in event_types

    @pytest.mark.asyncio
    async def test_error_event_flows_through_on_event(self):
        error_event = ErrorEvent(sequence_number=0, code="rate_limit", message="slow down")

        mw = TrackingMiddleware()
        await _run(mw, [error_event])

        event_types = [type(c[1]) for c in mw.calls if c[0] == "on_event"]
        assert ErrorEvent in event_types

    @pytest.mark.asyncio
    async def test_error_can_be_suppressed_via_on_event(self):
        class SuppressErrors(StreamMiddleware):
            async def on_event(self, ctx, event):
                if isinstance(event, ResponseFailedEvent):
                    return None
                return event

        events = await _run(SuppressErrors(), _error_backend_events())
        assert not any(isinstance(e, ResponseFailedEvent) for e in events)


# ---------------------------------------------------------------------------
# Tests: on_request
# ---------------------------------------------------------------------------


class TestOnRequest:
    @pytest.mark.asyncio
    async def test_on_request_can_modify_ctx(self):
        class InjectInstructions(StreamMiddleware):
            async def on_request(self, ctx):
                ctx.request = ctx.request.model_copy(update={"instructions": "be helpful"})

        mw = InjectInstructions()
        ctx = _make_ctx()
        chain = MiddlewareChain(middlewares=[mw], backend=_StaticBackend(_text_backend_events()))
        _ = [event async for event in chain.execute(ctx)]

        assert ctx.request.instructions == "be helpful"


# ---------------------------------------------------------------------------
# Tests: passthrough (no overrides)
# ---------------------------------------------------------------------------


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_default_streams_everything_through(self):
        mw = StreamMiddleware()
        backend_events = _text_backend_events("Hello world")
        events = await _run(mw, backend_events)

        assert any(isinstance(e, ResponseCreatedEvent) for e in events)
        assert any(isinstance(e, ResponseOutputTextDeltaEvent) for e in events)
        assert any(isinstance(e, ResponseOutputItemDoneEvent) for e in events)
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_output_items_in_completed_response(self):
        mw = StreamMiddleware()
        events = await _run(mw, _text_backend_events("hi"))

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        assert len(completed.response.output) == 1
        assert isinstance(completed.response.output[0], OutputMessage)


# ---------------------------------------------------------------------------
# Tests: nested chain execution (context isolation)
# ---------------------------------------------------------------------------


class TestContextIsolation:
    @pytest.mark.asyncio
    async def test_nested_execute_does_not_clobber_outer_state(self):
        """Simulates delegation: an inner chain.execute creates a new RequestContext
        without wiping the outer context's state."""

        class DelegatingMiddleware(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                ctx.state["outer_key"] = "outer_value"

                inner_ctx = _make_ctx("inner")
                inner_ctx.state["inner_key"] = "inner_value"
                inner_chain = MiddlewareChain(
                    middlewares=[StreamMiddleware()],
                    backend=_StaticBackend(_text_backend_events("inner")),
                )
                async for _ in inner_chain.execute(inner_ctx):
                    pass

                assert ctx.state.get("outer_key") == "outer_value"
                assert "inner_key" not in ctx.state
                return item

        mw = DelegatingMiddleware()
        ctx = _make_ctx()
        chain = MiddlewareChain(middlewares=[mw], backend=_StaticBackend(_text_backend_events("hi")))
        events = [event async for event in chain.execute(ctx)]

        assert any(isinstance(e, ResponseCompletedEvent) for e in events)
        assert ctx.state.get("outer_key") == "outer_value"

    @pytest.mark.asyncio
    async def test_request_state_contextvar_restored_after_nested_execute(self):
        """The ContextVar points back to the outer ctx's state after the inner chain finishes."""
        from subspace.middleware.context import request_state

        contextvar_after_inner: dict[str, Any] = {}

        class InnerExecuteMiddleware(StreamMiddleware):
            async def on_output_item_done(self, ctx, item):
                ctx.state["marker"] = "outer"

                inner_ctx = _make_ctx("inner")
                inner_chain = MiddlewareChain(
                    middlewares=[StreamMiddleware()],
                    backend=_StaticBackend(_text_backend_events("inner")),
                )
                async for _ in inner_chain.execute(inner_ctx):
                    pass

                contextvar_after_inner.update(request_state.get())
                return item

        mw = InnerExecuteMiddleware()
        ctx = _make_ctx()
        chain = MiddlewareChain(
            middlewares=[mw],
            backend=_StaticBackend(_text_backend_events("hi")),
        )
        async for _ in chain.execute(ctx):
            pass

        assert contextvar_after_inner.get("marker") == "outer"
