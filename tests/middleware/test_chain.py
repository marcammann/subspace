from collections.abc import AsyncIterator

import pytest
from conftest import StaticBackend, make_ctx, text_backend_events

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.chain import MiddlewareChain
from subspace.middleware.context import RequestContext
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    StreamEvent,
)


class TestExecute:
    @pytest.mark.anyio
    async def test_passthrough_yields_all_events(self):
        events_in = text_backend_events()
        chain = MiddlewareChain(middlewares=[], backend=StaticBackend(events_in))
        events_out = [e async for e in chain.execute(make_ctx())]
        assert len(events_out) == len(events_in)

    @pytest.mark.anyio
    async def test_output_items_on_completed(self):
        chain = MiddlewareChain(middlewares=[], backend=StaticBackend(text_backend_events()))
        events = [e async for e in chain.execute(make_ctx())]
        completed = next(e for e in events if isinstance(e, ResponseCompletedEvent))
        assert len(completed.response.output) == 1

    @pytest.mark.anyio
    async def test_output_items_on_incomplete(self):
        all_events = text_backend_events()
        incomplete_events = []
        for e in all_events:
            if isinstance(e, ResponseCompletedEvent):
                incomplete_events.append(
                    ResponseIncompleteEvent(
                        sequence_number=e.sequence_number,
                        response=e.response.model_copy(update={"status": "incomplete"}),
                    )
                )
            else:
                incomplete_events.append(e)

        chain = MiddlewareChain(middlewares=[], backend=StaticBackend(incomplete_events))
        events = [e async for e in chain.execute(make_ctx())]
        incomplete = next(e for e in events if isinstance(e, ResponseIncompleteEvent))
        assert len(incomplete.response.output) == 1


class TestMiddlewareOrdering:
    @pytest.mark.anyio
    async def test_middlewares_wrap_in_order(self):
        order = []

        class TagMiddleware(Middleware):
            def __init__(self, tag: str):
                self._tag = tag

            async def __call__(
                self, ctx: RequestContext, call_next: NextHandler
            ) -> AsyncIterator[StreamEvent]:
                order.append(f"{self._tag}_before")
                async for event in call_next(ctx):
                    yield event
                order.append(f"{self._tag}_after")

        chain = MiddlewareChain(
            middlewares=[TagMiddleware("outer"), TagMiddleware("inner")],
            backend=StaticBackend(text_backend_events()),
        )
        _ = [e async for e in chain.execute(make_ctx())]
        assert order == ["outer_before", "inner_before", "inner_after", "outer_after"]


class TestFinalize:
    @pytest.mark.anyio
    async def test_finalize_called_on_prepared_middlewares(self):
        finalized = []

        class TrackFinalize(Middleware):
            async def __call__(
                self, ctx: RequestContext, call_next: NextHandler
            ) -> AsyncIterator[StreamEvent]:
                async for event in call_next(ctx):
                    yield event

            async def finalize(self, ctx: RequestContext) -> None:
                finalized.append(True)

        mw = TrackFinalize()
        chain = MiddlewareChain(middlewares=[mw], backend=StaticBackend(text_backend_events()))
        _ = [e async for e in chain.execute(make_ctx())]
        assert len(finalized) == 1
