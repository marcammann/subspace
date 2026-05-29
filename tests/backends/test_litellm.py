from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from conftest import make_ctx

from subspace.backends.litellm import LitellmBackend, _build_messages
from subspace.models.common import Status
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseOutputTextDeltaEvent,
)
from subspace.models.items import (
    FunctionCall,
    FunctionCallOutput,
    InputMessage,
    OutputMessage,
)


class TestBuildMessages:
    def test_string_input(self):
        ctx = make_ctx(input="hello")
        msgs = _build_messages(ctx)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_with_instructions(self):
        ctx = make_ctx(input="hello", instructions="be helpful")
        msgs = _build_messages(ctx)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "be helpful"

    def test_function_call_grouping(self):
        items = [
            InputMessage(role="user", content="do stuff"),
            FunctionCall(
                id="fc_1", name="tool_a", call_id="call_1", arguments="{}", status=Status.COMPLETED
            ),
            FunctionCall(
                id="fc_2", name="tool_b", call_id="call_2", arguments="{}", status=Status.COMPLETED
            ),
            FunctionCallOutput(call_id="call_1", output="result_a"),
            FunctionCallOutput(call_id="call_2", output="result_b"),
        ]
        ctx = make_ctx(input=items)
        msgs = _build_messages(ctx)
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["tool_calls"]) == 2
        assert msgs[2]["role"] == "tool"
        assert msgs[3]["role"] == "tool"

    def test_output_message(self):
        items = [
            InputMessage(role="user", content="hi"),
            OutputMessage(
                id="msg_1",
                content=[OutputTextContent(text="hello back")],
                status=Status.COMPLETED,
            ),
        ]
        ctx = make_ctx(input=items)
        msgs = _build_messages(ctx)
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "hello back"


def _make_chunk(content: str | None = None, tool_calls=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta)
    chunk = SimpleNamespace(choices=[choice], usage=usage)
    return chunk


class _AsyncChunkIterator:
    def __init__(self, chunks):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class TestLitellmHandle:
    @pytest.mark.anyio
    async def test_text_stream(self):
        chunks = [_make_chunk(content="Hello"), _make_chunk(content=" world")]
        mock_stream = _AsyncChunkIterator(chunks)

        with patch("subspace.backends.litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream)
            backend = LitellmBackend(model="test-model")
            ctx = make_ctx()
            events = [e async for e in backend.handle(ctx)]

        assert any(isinstance(e, ResponseOutputTextDeltaEvent) for e in events)
        assert isinstance(events[-1], ResponseCompletedEvent)
        assert events[-1].response.status == Status.COMPLETED

    @pytest.mark.anyio
    async def test_error_yields_failed(self):
        with patch("subspace.backends.litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=RuntimeError("api error"))
            backend = LitellmBackend(model="test-model")
            ctx = make_ctx()
            events = [e async for e in backend.handle(ctx)]

        assert isinstance(events[-1], ResponseFailedEvent)
        assert events[-1].response.status == Status.FAILED
