from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.context import RequestContext, request_state
from subspace.models.events import StreamEvent

if TYPE_CHECKING:
    from subspace.backends.base import Backend


class MiddlewareChain:
    def __init__(self, middlewares: list[Middleware], backend: "Backend") -> None:
        self._middlewares = middlewares
        self._backend = backend

    async def execute(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        handler: NextHandler = self._backend.handle

        for mw in reversed(self._middlewares):
            handler = _wrap(mw, handler)

        token = request_state.set(ctx.state)
        try:
            async for event in handler(ctx):
                yield event
        finally:
            request_state.reset(token)
            await self.finalize(ctx)

    async def finalize(self, ctx: RequestContext) -> None:
        for mw in self._middlewares:
            if ctx.prepared(mw):
                await mw.finalize(ctx)


def _wrap(mw: Middleware, next_handler: NextHandler) -> NextHandler:
    async def wrapped(ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        if not ctx.prepared(mw):
            ctx.mark_prepared(mw)
            await mw.prepare(ctx)
        async for event in mw(ctx, next_handler):
            yield event

    return wrapped
