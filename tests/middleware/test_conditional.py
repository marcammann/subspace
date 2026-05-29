from collections.abc import AsyncIterator

import pytest
from conftest import StaticBackend, make_ctx, text_backend_events

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.chain import MiddlewareChain
from subspace.middleware.conditional import ConditionalMiddleware
from subspace.middleware.context import RequestContext
from subspace.models.events import StreamEvent


class TestConditionalMiddleware:
    @pytest.mark.anyio
    async def test_runs_when_should_run_true(self):
        called = []

        class Inner(Middleware):
            async def __call__(
                self, ctx: RequestContext, call_next: NextHandler
            ) -> AsyncIterator[StreamEvent]:
                called.append(True)
                async for event in call_next(ctx):
                    yield event

        chain = MiddlewareChain(
            middlewares=[ConditionalMiddleware(Inner())],
            backend=StaticBackend(text_backend_events()),
        )
        _ = [e async for e in chain.execute(make_ctx())]
        assert len(called) == 1

    @pytest.mark.anyio
    async def test_skips_when_should_run_false(self):
        called = []

        class Inner(Middleware):
            async def __call__(
                self, ctx: RequestContext, call_next: NextHandler
            ) -> AsyncIterator[StreamEvent]:
                called.append(True)
                async for event in call_next(ctx):
                    yield event

        class NeverRun(ConditionalMiddleware):
            async def should_run(self, ctx: RequestContext) -> bool:
                return False

        chain = MiddlewareChain(
            middlewares=[NeverRun(Inner())],
            backend=StaticBackend(text_backend_events()),
        )
        events = [e async for e in chain.execute(make_ctx())]
        assert len(called) == 0
        assert len(events) > 0
