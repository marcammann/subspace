from collections.abc import AsyncIterator
from typing import Any

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.common import Status, Usage
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseOutputItemDoneEvent,
    StreamEvent,
)
from subspace.models.items import FunctionCall, OutputMessage, ServerFunctionCall
from subspace.models.response import ResponseResource


class StreamMiddleware(Middleware):
    """Callback-driven middleware base class for reacting to stream events.

    Subclass and override callbacks to react to the stream without
    managing the event plumbing yourself.

    Three modes per output item:
    - Stream through: pass events via on_event, return item from on_output_item_done
    - Buffer then decide: suppress events via on_event → None, inspect in on_output_item_done
    - Drop entirely: suppress via on_event → None, return None from on_output_item_done
    """

    async def on_request(self, ctx: RequestContext) -> None:
        """Called once per request, before streaming starts. Modify ctx.request here."""

    async def prepare(self, ctx: RequestContext) -> None:
        await self.on_request(ctx)

    # NOTE: on_event currently sees all events except ResponseCompletedEvent (handled by
    # on_response_completed). ResponseOutputItemDoneEvent flows through both on_output_item_done
    # AND on_event. Need to decide: should on_event be the single gate for ALL events, or should
    # specialized callbacks (on_output_item_done, on_response_completed) fully own their events?
    async def on_event(self, ctx: RequestContext, event: StreamEvent) -> StreamEvent | None:
        """Called for every event. Return None to suppress it from the client stream."""
        return event

    async def on_output_item_done(
        self,
        ctx: RequestContext,
        item: OutputMessage | FunctionCall | ServerFunctionCall,
    ) -> OutputMessage | FunctionCall | ServerFunctionCall | None:
        """Called when an output item is fully assembled (response.output_item.done).

        Return the item (possibly transformed) to include it in the output.
        Return None to drop it from the final response.
        """
        return item

    async def on_response_completed(
        self, ctx: RequestContext, response: ResponseResource
    ) -> ResponseResource | None:
        """Called when the response is assembled (response.completed).

        Return the response to emit the CompletedEvent (possibly transformed).
        Return None to suppress it.
        """
        return response

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        output_items: list[Any] = []
        total_usage = Usage()
        seq = 0

        async for event in call_next(ctx):
            seq = max(seq, event.sequence_number) + 1

            if isinstance(event, ResponseCompletedEvent):
                if event.response.usage:
                    u = event.response.usage
                    total_usage = Usage(
                        input_tokens=total_usage.input_tokens + u.input_tokens,
                        output_tokens=total_usage.output_tokens + u.output_tokens,
                        total_tokens=total_usage.total_tokens + u.total_tokens,
                    )

                continue

            if isinstance(event, ResponseOutputItemDoneEvent):
                result = await self.on_output_item_done(ctx, event.item)
                if result is None:
                    continue

                output_items.append(result)
                if result is not event.item:
                    event = event.model_copy(update={"item": result})

            dispatched = await self.on_event(ctx, event)
            if dispatched is not None:
                yield dispatched

        final = ctx.response.model_copy(
            update={
                "status": Status.COMPLETED,
                "output": output_items,
                "usage": total_usage,
            }
        )

        final = await self.on_response_completed(ctx, final)
        if final is not None:
            yield ResponseCompletedEvent(sequence_number=seq, response=final)
