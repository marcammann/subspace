"""Middleware that can retract streamed responses.

Extends StreamTapMiddleware with intervention capability: when consume()
returns events, the upstream is cancelled, open items are closed, the
returned events are emitted, and the response completes as incomplete.

    class ToxicityGuard(RetractionMiddleware):
        @staticmethod
        def local_factory():
            return {"text": "", "chars": 0}

        def should_dispatch(self, ctx, event):
            if not isinstance(event, ResponseOutputTextDeltaEvent):
                return False
            local = ctx.local(self)
            local["text"] += event.delta
            local["chars"] += len(event.delta)
            if local["chars"] >= 100:
                local["chars"] = 0
                return True
            return False

        async def consume(self, ctx, event):
            score = await toxicity_api(ctx.local(self)["text"])
            if score > 0.8:
                return ResponseOutputTextRetractedEvent(
                    sequence_number=0, reason="toxic content",
                )
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from subspace.middleware.base import NextHandler
from subspace.middleware.context import RequestContext
from subspace.middleware.stream_tap import StreamTapMiddleware
from subspace.middleware.utils import StreamTracker
from subspace.models.events import (
    BaseStreamEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    StreamEvent,
)


class ResponseOutputTextRetractedEvent(BaseStreamEvent):
    type: Literal["response.output_text.retracted"] = "response.output_text.retracted"
    item_id: str | None = None
    output_index: int | None = None
    reason: str | None = None


class RetractionMiddleware(StreamTapMiddleware):
    """StreamTapMiddleware with intervention support.

    Override should_dispatch() and consume(). Return event(s) from
    consume() to intervene — the upstream is cancelled, open items are
    closed, your events are emitted, and the response ends as incomplete.

    consume() calls run concurrently — the first one that returns a
    non-None result triggers intervention.
    """

    async def consume(  # type: ignore[override]
        self, ctx: RequestContext, event: StreamEvent
    ) -> StreamEvent | list[StreamEvent] | None:
        return None

    async def __call__(
        self, ctx: RequestContext, call_next: NextHandler
    ) -> AsyncIterator[StreamEvent]:
        intervention_events: list[StreamEvent] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)
        pending: set[asyncio.Task] = set()
        tracker = StreamTracker()

        def _collect(task: asyncio.Task) -> None:
            pending.discard(task)
            if intervention_events:
                return
            if task.cancelled() or task.exception():
                return
            result = task.result()
            if result is None:
                return
            if isinstance(result, list):
                intervention_events.extend(result)
            else:
                intervention_events.append(result)

        async def _bounded_consume(event: StreamEvent) -> StreamEvent | list[StreamEvent] | None:
            async with semaphore:
                return await self.consume(ctx, event)

        try:
            async for event in call_next(ctx):
                tracker.track(event)

                if intervention_events:
                    break

                if isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent)):
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    if intervention_events:
                        break
                    yield event
                    return

                yield event

                if self.should_dispatch(ctx, event):
                    task = asyncio.create_task(_bounded_consume(event))
                    pending.add(task)
                    task.add_done_callback(_collect)

            if intervention_events:
                item_id = tracker.last_text_item_id
                output_index = tracker.last_text_output_index

                for close_event in tracker.close_open_items():
                    yield close_event

                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                for evt in intervention_events:
                    updates: dict[str, object] = {}
                    if isinstance(evt, BaseStreamEvent):
                        updates["sequence_number"] = tracker.next_seq()
                    if item_id and isinstance(evt, ResponseOutputTextRetractedEvent):
                        updates["item_id"] = item_id
                        updates["output_index"] = output_index
                    if updates:
                        evt = evt.model_copy(update=updates)
                    yield evt

                assert tracker.response is not None
                response = tracker.response.model_copy(
                    update={"status": "incomplete"}
                )
                yield ResponseIncompleteEvent(
                    sequence_number=tracker.next_seq(), response=response
                )

        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
