import json

import pytest
from conftest import text_backend_events

from subspace.streaming.sse import stream_to_sse


async def _to_list(events):
    async def _gen():
        for e in events:
            yield e

    return [item async for item in stream_to_sse(_gen())]


class TestStreamToSSE:
    @pytest.mark.anyio
    async def test_yields_event_dicts(self):
        events = text_backend_events("hi")
        result = await _to_list(events)
        for item in result[:-1]:
            assert "event" in item
            assert "data" in item

    @pytest.mark.anyio
    async def test_ends_with_done(self):
        result = await _to_list(text_backend_events("hi"))
        assert result[-1] == {"data": "[DONE]"}

    @pytest.mark.anyio
    async def test_data_is_valid_json(self):
        result = await _to_list(text_backend_events("hi"))
        for item in result[:-1]:
            parsed = json.loads(item["data"])
            assert "type" in parsed
