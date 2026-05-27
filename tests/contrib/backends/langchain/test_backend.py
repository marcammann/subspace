import time
from types import SimpleNamespace
from typing import Any

import pytest

from subspace.contrib.backends.langchain.backend import (
    LangchainBackend,
    _build_messages,
    _extract_text_parts,
    _make_config,
    _SequenceCounter,
    _TextBuffer,
    _ToolCallBuffer,
)
from subspace.middleware.context import RequestContext
from subspace.models.common import Status
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgsDeltaEvent,
    ResponseFunctionCallArgsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
)
from subspace.models.items import FunctionCall, InputMessage, OutputMessage
from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    input: str | list = "hello",
    instructions: str | None = None,
    metadata: dict | None = None,
) -> RequestContext:
    request = CreateResponseRequest(model="test", input=input, instructions=instructions)
    response = ResponseResource(
        id="resp_test",
        created_at=int(time.time()),
        status=Status.IN_PROGRESS,
        model="test",
    )
    return RequestContext(
        request=request,
        response_id="resp_test",
        response=response,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# _extract_text_parts
# ---------------------------------------------------------------------------


class TestExtractTextParts:
    def test_string_content(self):
        chunk = SimpleNamespace(content="hello")
        assert _extract_text_parts(chunk) == ["hello"]

    def test_empty_string(self):
        chunk = SimpleNamespace(content="")
        assert _extract_text_parts(chunk) == []

    def test_list_of_text_blocks(self):
        chunk = SimpleNamespace(content=[
            {"type": "text", "text": "hello"},
            {"type": "text", "text": " world"},
        ])
        assert _extract_text_parts(chunk) == ["hello", " world"]

    def test_list_with_non_text_blocks(self):
        chunk = SimpleNamespace(content=[
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "123"},
        ])
        assert _extract_text_parts(chunk) == ["hello"]

    def test_list_of_raw_strings(self):
        chunk = SimpleNamespace(content=["hello", " world"])
        assert _extract_text_parts(chunk) == ["hello", " world"]

    def test_empty_text_block_skipped(self):
        chunk = SimpleNamespace(content=[
            {"type": "text", "text": ""},
            {"type": "text", "text": "real"},
        ])
        assert _extract_text_parts(chunk) == ["real"]

    def test_none_content(self):
        chunk = SimpleNamespace(content=None)
        assert _extract_text_parts(chunk) == []

    def test_mixed_strings_and_dicts(self):
        chunk = SimpleNamespace(content=[
            "plain",
            {"type": "text", "text": "block"},
        ])
        assert _extract_text_parts(chunk) == ["plain", "block"]


# ---------------------------------------------------------------------------
# _SequenceCounter
# ---------------------------------------------------------------------------


class TestSequenceCounter:
    def test_starts_at_zero(self):
        seq = _SequenceCounter()
        assert seq.current == 0

    def test_next_increments(self):
        seq = _SequenceCounter()
        assert seq.next() == 0
        assert seq.next() == 1
        assert seq.current == 2

    def test_custom_start(self):
        seq = _SequenceCounter(start=10)
        assert seq.next() == 10


# ---------------------------------------------------------------------------
# _TextBuffer
# ---------------------------------------------------------------------------


class TestTextBuffer:
    def test_first_delta_emits_three_events(self):
        buf = _TextBuffer()
        seq = _SequenceCounter()
        events = buf.delta("hello", seq, output_items=[])
        assert len(events) == 3
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
        assert isinstance(events[1], ResponseContentPartAddedEvent)
        assert isinstance(events[2], ResponseOutputTextDeltaEvent)
        assert events[2].delta == "hello"

    def test_second_delta_emits_one_event(self):
        buf = _TextBuffer()
        seq = _SequenceCounter()
        buf.delta("hello", seq, output_items=[])
        events = buf.delta(" world", seq, output_items=[])
        assert len(events) == 1
        assert isinstance(events[0], ResponseOutputTextDeltaEvent)
        assert events[0].delta == " world"

    def test_flush_emits_done_events(self):
        buf = _TextBuffer()
        seq = _SequenceCounter()
        items: list[Any] = []
        buf.delta("hello world", seq, output_items=items)
        events = buf.flush(seq, output_items=items)
        assert len(events) == 3
        assert isinstance(events[0], ResponseOutputTextDoneEvent)
        assert events[0].text == "hello world"
        assert isinstance(events[2], ResponseOutputItemDoneEvent)
        assert len(items) == 1
        assert isinstance(items[0], OutputMessage)
        assert items[0].status == Status.COMPLETED

    def test_flush_noop_when_inactive(self):
        buf = _TextBuffer()
        seq = _SequenceCounter()
        assert buf.flush(seq, output_items=[]) == []

    def test_sequence_numbers_increment(self):
        buf = _TextBuffer()
        seq = _SequenceCounter(start=5)
        events = buf.delta("hi", seq, output_items=[])
        assert events[0].sequence_number == 5
        assert events[1].sequence_number == 6
        assert events[2].sequence_number == 7

    def test_output_index_tracks_items(self):
        buf = _TextBuffer()
        seq = _SequenceCounter()
        existing = [object(), object()]
        events = buf.delta("hi", seq, output_items=existing)
        assert events[0].output_index == 2


# ---------------------------------------------------------------------------
# _ToolCallBuffer
# ---------------------------------------------------------------------------


class TestToolCallBuffer:
    def test_emits_events_on_first_chunk(self):
        buf = _ToolCallBuffer()
        seq = _SequenceCounter()
        items: list[Any] = []

        events = buf.chunk(
            {"index": 0, "id": "call_123", "name": "my_tool", "args": '{"x":'},
            seq, items,
        )
        assert len(events) == 2
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
        assert isinstance(events[0].item, FunctionCall)
        assert events[0].item.name == "my_tool"
        assert isinstance(events[1], ResponseFunctionCallArgsDeltaEvent)
        assert events[1].delta == '{"x":'

    def test_accumulates_args(self):
        buf = _ToolCallBuffer()
        seq = _SequenceCounter()
        items: list[Any] = []

        buf.chunk({"index": 0, "id": "call_1", "name": "t", "args": '{"a":'}, seq, items)
        events = buf.chunk({"index": 0, "args": '"b"}'}, seq, items)

        assert len(events) == 1
        assert isinstance(events[0], ResponseFunctionCallArgsDeltaEvent)
        assert events[0].delta == '"b"}'

    def test_flush(self):
        buf = _ToolCallBuffer()
        seq = _SequenceCounter()
        items: list[Any] = []

        buf.chunk({"index": 0, "id": "call_1", "name": "t", "args": '{"a": 1}'}, seq, items)
        events = buf.flush(seq, items)

        assert len(events) == 2
        assert isinstance(events[0], ResponseFunctionCallArgsDoneEvent)
        assert events[0].arguments == '{"a": 1}'
        assert isinstance(events[1], ResponseOutputItemDoneEvent)
        assert isinstance(events[1].item, FunctionCall)
        assert events[1].item.status == Status.COMPLETED
        assert len(items) == 1

    def test_flush_clears_state(self):
        buf = _ToolCallBuffer()
        seq = _SequenceCounter()
        items: list[Any] = []

        buf.chunk({"index": 0, "id": "call_1", "name": "t", "args": "{}"}, seq, items)
        buf.flush(seq, items)

        assert buf.flush(seq, items) == []

    def test_multiple_tools(self):
        buf = _ToolCallBuffer()
        seq = _SequenceCounter()
        items: list[Any] = []

        buf.chunk({"index": 0, "id": "call_a", "name": "tool_a", "args": "{}"}, seq, items)
        buf.chunk({"index": 1, "id": "call_b", "name": "tool_b", "args": "{}"}, seq, items)

        events = buf.flush(seq, items)
        assert len(events) == 4  # done + item_done for each
        assert len(items) == 2


# ---------------------------------------------------------------------------
# _make_config
# ---------------------------------------------------------------------------


class TestMakeConfig:
    def test_thread_id_from_request_metadata(self):
        ctx = _make_ctx(metadata={})
        ctx.request = ctx.request.model_copy(update={"metadata": {"thread_id": "from-request"}})
        config = _make_config(ctx)
        assert config["configurable"]["thread_id"] == "from-request"

    def test_thread_id_from_ctx_metadata(self):
        ctx = _make_ctx(metadata={"thread_id": "from-ctx"})
        config = _make_config(ctx)
        assert config["configurable"]["thread_id"] == "from-ctx"

    def test_thread_id_fallback_to_response_id(self):
        ctx = _make_ctx()
        config = _make_config(ctx)
        assert config["configurable"]["thread_id"] == "resp_test"


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_string_input(self):
        ctx = _make_ctx(input="hello")
        msgs = _build_messages(ctx)
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_string_input_with_instructions(self):
        ctx = _make_ctx(input="hello", instructions="be helpful")
        msgs = _build_messages(ctx)
        assert len(msgs) == 2
        assert msgs[0].content == "be helpful"
        assert msgs[1].content == "hello"

    def test_empty_string_input(self):
        ctx = _make_ctx(input="")
        msgs = _build_messages(ctx)
        assert len(msgs) == 0

    def test_item_list_input(self):
        items = [
            InputMessage(role="user", content="question"),
            OutputMessage(
                id="msg_1",
                content=[OutputTextContent(text="answer")],
                status=Status.COMPLETED,
            ),
            InputMessage(role="user", content="follow-up"),
        ]
        ctx = _make_ctx(input=items)
        msgs = _build_messages(ctx)
        assert len(msgs) == 3
        assert msgs[0].content == "question"
        assert msgs[1].content == "answer"
        assert msgs[2].content == "follow-up"


# ---------------------------------------------------------------------------
# LangchainBackend.handle
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal mock that yields AIMessageChunks via astream."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def astream(self, input, *, config, stream_mode, version):
        from langchain_core.messages import AIMessageChunk

        for text in self._chunks:
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content=text), {}),
            }


class TestLangchainBackendHandle:
    @pytest.mark.asyncio
    async def test_streams_text(self):
        graph = _FakeGraph(["Hello", " world"])
        backend = LangchainBackend(graph)
        ctx = _make_ctx()

        events = [e async for e in backend.handle(ctx)]
        types = [type(e) for e in events]

        assert types[0] is ResponseCreatedEvent
        assert types[1] is ResponseInProgressEvent
        assert ResponseOutputTextDeltaEvent in types
        assert types[-1] is ResponseCompletedEvent
        assert events[-1].response.status == Status.COMPLETED

    @pytest.mark.asyncio
    async def test_factory_receives_ctx_and_tools(self):
        received = []

        def factory(ctx, interrupt_tools):
            received.append((ctx, interrupt_tools))
            return _FakeGraph(["ok"])

        backend = LangchainBackend(factory)
        ctx = _make_ctx()
        _ = [e async for e in backend.handle(ctx)]
        assert len(received) == 1
        assert received[0][0] is ctx
        assert isinstance(received[0][1], list)

    @pytest.mark.asyncio
    async def test_error_yields_failed(self):
        class _BrokenGraph:
            async def astream(self, *args, **kwargs):
                raise RuntimeError("boom")
                yield  # noqa: B027

        backend = LangchainBackend(_BrokenGraph())
        ctx = _make_ctx()
        events = [e async for e in backend.handle(ctx)]
        assert isinstance(events[-1], ResponseFailedEvent)
        assert events[-1].response.status == Status.FAILED

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        backend = LangchainBackend(_FakeGraph([]))
        ctx = _make_ctx()
        events = [e async for e in backend.handle(ctx)]
        types = [type(e) for e in events]
        assert types == [ResponseCreatedEvent, ResponseInProgressEvent, ResponseCompletedEvent]
