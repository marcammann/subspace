from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from subspace.middleware.base import NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.events import ResponseCompletedEvent, StreamEvent

if TYPE_CHECKING:
    from subspace.middleware.conversation_history.storage.base import Storage


class ConversationHistoryMiddleware:
    """Enables multi-turn conversations via previous_response_id.

    On the way in: loads prior input/output items from storage and prepends them
    to the current request input. On the way out: persists the completed response
    so future requests can reference it.
    """

    def __init__(self, storage: "Storage") -> None:
        self._storage = storage

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        if ctx.request.previous_response_id:
            history = await self._storage.load_history(ctx.request.previous_response_id)
            if history:
                ctx.request = ctx.request.model_copy(
                    update={"input": history + ctx.request.input}
                )

        async for event in call_next(ctx):
            yield event

            if isinstance(event, ResponseCompletedEvent):
                await self._storage.save_response(
                    ctx.response_id,
                    ctx.request.input,
                    list(event.response.output),
                )
