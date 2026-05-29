from collections.abc import AsyncIterator

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.events import StreamEvent


class ConditionalMiddleware(Middleware):
    """Wraps a middleware and conditionally skips it based on request inspection.

    Subclass and override `should_run` to control when the wrapped middleware executes.
    When skipped, the request passes straight through to the next middleware in the chain.

        class AuthRequired(ConditionalMiddleware):
            async def should_run(self, ctx: RequestContext) -> bool:
                return ctx.metadata.get("authenticated", False)

        chain_mw = AuthRequired(expensive_middleware)
    """

    def __init__(self, middleware: Middleware) -> None:
        self._middleware = middleware

    async def should_run(self, ctx: RequestContext) -> bool:
        return True

    async def __aenter__(self):
        await self._middleware.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._middleware.__aexit__(exc_type, exc_val, exc_tb)

    async def prepare(self, ctx: RequestContext) -> None:
        should_run = await self.should_run(ctx)
        ctx.local(self)["should_run"] = should_run
        if should_run:
            await self._middleware.prepare(ctx)

    async def finalize(self, ctx: RequestContext) -> None:
        if ctx.local(self).get("should_run", False):
            await self._middleware.finalize(ctx)

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        if ctx.local(self).get("should_run", False):
            async for event in self._middleware(ctx, call_next):
                yield event
        else:
            async for event in call_next(ctx):
                yield event
