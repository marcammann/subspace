from collections.abc import AsyncIterator, Callable

from subspace.middleware.context import RequestContext
from subspace.models.agent import AgentCapabilities
from subspace.models.events import StreamEvent

type NextHandler = Callable[[RequestContext], AsyncIterator[StreamEvent]]


class Middleware:
    local_factory: type = dict

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def prepare(self, ctx: RequestContext) -> None:
        """Called once per request, before streaming starts. Modify ctx.request here."""

    async def finalize(self, ctx: RequestContext) -> None:
        """Called once per request, after the stream ends. Clean up resources here."""

    def transform_capabilities(self, capabilities: AgentCapabilities) -> AgentCapabilities:
        """Return capabilities after this middleware is applied to a backend."""
        return capabilities

    def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
