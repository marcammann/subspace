from collections.abc import AsyncIterator

from subspace.middleware.base import NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.events import ResponseCompletedEvent, StreamEvent


class StreamAggregatorMiddleware:
    """Buffers the entire stream and emits only the final ResponseCompletedEvent.

    Used to support non-streaming responses: the internal transport is always streaming,
    and this middleware collapses it into a single completed response for clients that
    set stream=false.
    """

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        last_completed: ResponseCompletedEvent | None = None

        async for event in call_next(ctx):
            if isinstance(event, ResponseCompletedEvent):
                last_completed = event

        if last_completed:
            yield last_completed
