from collections.abc import AsyncIterator

from subspace.models.events import StreamEvent


async def stream_to_sse(
    events: AsyncIterator[StreamEvent],
) -> AsyncIterator[dict[str, str]]:
    async for event in events:
        yield {
            "event": event.type,
            "data": event.model_dump_json(),
        }
    yield {"data": "[DONE]"}
