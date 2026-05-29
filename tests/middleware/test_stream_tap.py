import pytest
from conftest import StaticBackend, make_ctx, text_backend_events

from subspace.contrib.middleware.retraction import (
    ResponseOutputTextRetractedEvent,
    RetractionMiddleware,
)
from subspace.middleware.chain import MiddlewareChain
from subspace.middleware.context import RequestContext
from subspace.middleware.stream_tap import StreamTapMiddleware
from subspace.models.common import Status
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponseOutputTextDeltaEvent,
    StreamEvent,
)


class RetractOnKeyword(RetractionMiddleware):
    """Retracts when accumulated text contains 'BADWORD'."""

    @staticmethod
    def local_factory() -> dict:
        return {"text": ""}

    def should_dispatch(self, ctx: RequestContext, event: StreamEvent) -> bool:
        if not isinstance(event, ResponseOutputTextDeltaEvent):
            return False
        ctx.local(self)["text"] += event.delta
        return True

    async def consume(
        self, ctx: RequestContext, event: StreamEvent
    ) -> StreamEvent | list[StreamEvent] | None:
        text = ctx.local(self)["text"]
        if "BADWORD" in text:
            return ResponseOutputTextRetractedEvent(
                sequence_number=0, reason="contains BADWORD"
            )
        return None


class TestStreamTapPassthrough:
    @pytest.mark.anyio
    async def test_no_intervention_passes_all_events(self):
        mw = StreamTapMiddleware()
        ctx = make_ctx()
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(text_backend_events()))
        events = [e async for e in chain.execute(ctx)]

        types = [e.type for e in events]
        assert "response.created" in types
        assert "response.output_text.delta" in types
        assert "response.completed" in types


class TestStreamTapIntervention:
    @pytest.mark.anyio
    async def test_retraction_event_emitted(self):
        mw = RetractOnKeyword()
        ctx = make_ctx()
        backend_events = text_backend_events("This has a BADWORD in it")
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(backend_events))
        events = [e async for e in chain.execute(ctx)]

        types = [e.type for e in events]
        assert "response.output_text.retracted" in types, (
            f"Expected retraction event but got: {types}"
        )

    @pytest.mark.anyio
    async def test_retraction_before_incomplete(self):
        mw = RetractOnKeyword()
        ctx = make_ctx()
        backend_events = text_backend_events("This has a BADWORD in it")
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(backend_events))
        events = [e async for e in chain.execute(ctx)]

        types = [e.type for e in events]
        retract_idx = types.index("response.output_text.retracted")
        incomplete_idx = types.index("response.incomplete")
        assert retract_idx < incomplete_idx

    @pytest.mark.anyio
    async def test_incomplete_event_on_intervention(self):
        mw = RetractOnKeyword()
        ctx = make_ctx()
        backend_events = text_backend_events("This has a BADWORD in it")
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(backend_events))
        events = [e async for e in chain.execute(ctx)]

        assert any(isinstance(e, ResponseIncompleteEvent) for e in events)
        assert not any(isinstance(e, ResponseCompletedEvent) for e in events)

        incomplete = next(e for e in events if isinstance(e, ResponseIncompleteEvent))
        assert incomplete.response.status == Status.INCOMPLETE
        assert len(incomplete.response.output) > 0

        retraction = next(
            e for e in events if isinstance(e, ResponseOutputTextRetractedEvent)
        )
        assert retraction.reason == "contains BADWORD"
        assert retraction.item_id is not None
        assert retraction.output_index == 0

    @pytest.mark.anyio
    async def test_no_retraction_without_keyword(self):
        mw = RetractOnKeyword()
        ctx = make_ctx()
        backend_events = text_backend_events("This is perfectly fine")
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(backend_events))
        events = [e async for e in chain.execute(ctx)]

        assert not any(isinstance(e, ResponseOutputTextRetractedEvent) for e in events)
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)
