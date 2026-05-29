"""Middleware that taps into the stream without blocking it.

Events flow through to the client immediately. Each dispatched event
spawns its own consume() task, so multiple events are processed
concurrently (up to max_concurrency).

    class AnalyticsObserver(StreamTapMiddleware):
        max_concurrency = 20

        def should_dispatch(self, ctx, event):
            return True

        async def consume(self, ctx, event):
            await analytics.track(event)
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from subspace.middleware.base import Middleware, NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    StreamEvent,
)

logger = logging.getLogger(__name__)


class StreamTapMiddleware(Middleware):
    """Dispatches stream events to concurrent background consumers.

    Override should_dispatch() to control which events are dispatched,
    and consume() to process them asynchronously. Each dispatched event
    gets its own task — consume() calls run concurrently, bounded by
    max_concurrency.
    """

    def __init__(self, *, max_concurrency: int = 10) -> None:
        self.max_concurrency = max_concurrency

    def should_dispatch(self, ctx: RequestContext, event: StreamEvent) -> bool:
        return False

    async def consume(self, ctx: RequestContext, event: StreamEvent) -> None:
        pass

    async def __call__(
        self, ctx: RequestContext, call_next: NextHandler
    ) -> AsyncIterator[StreamEvent]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        pending: set[asyncio.Task] = set()

        async def _bounded_consume(event: StreamEvent) -> None:
            async with semaphore:
                await self.consume(ctx, event)

        async for event in call_next(ctx):
            if isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent)):
                if pending:
                    results = await asyncio.gather(*pending, return_exceptions=True)
                    _log_consumer_errors(results)
                yield event
                return

            yield event

            if self.should_dispatch(ctx, event):
                task = asyncio.create_task(_bounded_consume(event))
                pending.add(task)
                task.add_done_callback(pending.discard)

        if pending:
            results = await asyncio.gather(*pending, return_exceptions=True)
            _log_consumer_errors(results)


def _log_consumer_errors(results: list[object]) -> None:
    for result in results:
        if isinstance(result, Exception):
            logger.error(
                "Stream tap consumer failed",
                exc_info=(type(result), result, result.__traceback__),
            )
