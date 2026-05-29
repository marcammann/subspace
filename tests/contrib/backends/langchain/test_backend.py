import pytest
from conftest import make_ctx

from subspace.contrib.backends.langchain.backend import (
    LangchainBackend,
    _build_messages,
    _build_stream_input,
    _extract_text_parts,
    _make_config,
)
from subspace.models.common import Status
from subspace.models.content import OutputTextContent
from subspace.models.events import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
)
from subspace.models.items import (
    FunctionCall,
    FunctionCallOutput,
    InputMessage,
    OutputMessage,
    ServerFunctionCall,
)

# ---------------------------------------------------------------------------
# _extract_text_parts
# ---------------------------------------------------------------------------


class TestExtractTextParts:
    def test_string_content(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(content="hello")
        assert _extract_text_parts(chunk) == ["hello"]

    def test_empty_string(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(content="")
        assert _extract_text_parts(chunk) == []

    def test_list_of_text_blocks(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"},
            ]
        )
        assert _extract_text_parts(chunk) == ["hello", " world"]

    def test_list_with_non_text_blocks(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "123"},
            ]
        )
        assert _extract_text_parts(chunk) == ["hello"]

    def test_list_of_raw_strings(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(content=["hello", " world"])
        assert _extract_text_parts(chunk) == ["hello", " world"]

    def test_empty_text_block_skipped(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(
            content=[
                {"type": "text", "text": ""},
                {"type": "text", "text": "real"},
            ]
        )
        assert _extract_text_parts(chunk) == ["real"]

    def test_none_content(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(content=None)
        assert _extract_text_parts(chunk) == []

    def test_mixed_strings_and_dicts(self):
        from types import SimpleNamespace

        chunk = SimpleNamespace(
            content=[
                "plain",
                {"type": "text", "text": "block"},
            ]
        )
        assert _extract_text_parts(chunk) == ["plain", "block"]


# ---------------------------------------------------------------------------
# _make_config
# ---------------------------------------------------------------------------


class TestMakeConfig:
    def test_thread_id_from_request_metadata(self):
        ctx = make_ctx(metadata={})
        ctx.request = ctx.request.model_copy(update={"metadata": {"thread_id": "from-request"}})
        config = _make_config(ctx)
        assert config["configurable"]["thread_id"] == "from-request"

    def test_thread_id_from_ctx_metadata(self):
        ctx = make_ctx(metadata={"thread_id": "from-ctx"})
        config = _make_config(ctx)
        assert config["configurable"]["thread_id"] == "from-ctx"

    def test_thread_id_fallback_to_response_id(self):
        ctx = make_ctx()
        config = _make_config(ctx)
        assert config["configurable"]["thread_id"] == "resp_test"


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_string_input(self):
        ctx = make_ctx(input="hello")
        msgs = _build_messages(ctx)
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_string_input_with_instructions(self):
        ctx = make_ctx(input="hello", instructions="be helpful")
        msgs = _build_messages(ctx)
        assert len(msgs) == 2
        assert msgs[0].content == "be helpful"
        assert msgs[1].content == "hello"

    def test_empty_string_input(self):
        ctx = make_ctx(input="")
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
        ctx = make_ctx(input=items)
        msgs = _build_messages(ctx)
        assert len(msgs) == 3
        assert msgs[0].content == "question"
        assert msgs[1].content == "answer"
        assert msgs[2].content == "follow-up"

    def test_function_call_and_output_messages(self):
        items = [
            InputMessage(role="user", content="question"),
            FunctionCall(
                id="fc_1",
                name="lookup",
                call_id="call_1",
                arguments='{"query": "x"}',
                status=Status.COMPLETED,
            ),
            FunctionCallOutput(call_id="call_1", output="result"),
        ]
        ctx = make_ctx(input=items)

        msgs = _build_messages(ctx)

        assert len(msgs) == 3
        assert msgs[1].tool_calls == [
            {"name": "lookup", "args": {"query": "x"}, "id": "call_1", "type": "tool_call"}
        ]
        assert msgs[2].tool_call_id == "call_1"
        assert msgs[2].content == "result"


# ---------------------------------------------------------------------------
# _build_stream_input
# ---------------------------------------------------------------------------


class TestBuildStreamInput:
    def test_messages_input_without_resume_state(self):
        ctx = make_ctx(input="hello")
        stream_input = _build_stream_input(ctx)
        assert "messages" in stream_input
        assert stream_input["messages"][0].content == "hello"

    def test_command_resume_from_trailing_tool_outputs(self):
        ctx = make_ctx(
            input=[
                InputMessage(role="user", content="question"),
                FunctionCallOutput(call_id="call_1", output="one"),
                FunctionCallOutput(call_id="call_2", output="two"),
            ]
        )
        ctx.state["_interrupt_map"] = {"call_1": "intr_1", "call_2": "intr_2"}

        stream_input = _build_stream_input(ctx)

        assert stream_input.resume == {"intr_1": "one", "intr_2": "two"}
        assert "_interrupt_map" not in ctx.state


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


class _FakePartsGraph:
    def __init__(self, parts: list[dict]) -> None:
        self._parts = parts

    async def astream(self, input, *, config, stream_mode, version):
        for part in self._parts:
            yield part


class TestLangchainBackendHandle:
    @pytest.mark.anyio
    async def test_streams_text(self):
        graph = _FakeGraph(["Hello", " world"])
        backend = LangchainBackend(graph)
        ctx = make_ctx()

        events = [e async for e in backend.handle(ctx)]
        types = [type(e) for e in events]

        assert types[0] is ResponseCreatedEvent
        assert types[1] is ResponseInProgressEvent
        assert ResponseOutputTextDeltaEvent in types
        assert types[-1] is ResponseCompletedEvent
        assert events[-1].response.status == Status.COMPLETED

    @pytest.mark.anyio
    async def test_factory_receives_ctx_and_tools(self):
        received = []

        def factory(ctx, interrupt_tools):
            received.append((ctx, interrupt_tools))
            return _FakeGraph(["ok"])

        backend = LangchainBackend(factory)
        ctx = make_ctx()
        _ = [e async for e in backend.handle(ctx)]
        assert len(received) == 1
        assert received[0][0] is ctx
        assert isinstance(received[0][1], list)

    @pytest.mark.anyio
    async def test_error_yields_failed(self):
        class _BrokenGraph:
            async def astream(self, *args, **kwargs):
                raise RuntimeError("boom")
                yield  # noqa: B027

        backend = LangchainBackend(_BrokenGraph())
        ctx = make_ctx()
        events = [e async for e in backend.handle(ctx)]
        assert isinstance(events[-1], ResponseFailedEvent)
        assert events[-1].response.status == Status.FAILED

    @pytest.mark.anyio
    async def test_empty_stream(self):
        backend = LangchainBackend(_FakeGraph([]))
        ctx = make_ctx()
        events = [e async for e in backend.handle(ctx)]
        types = [type(e) for e in events]
        assert types == [ResponseCreatedEvent, ResponseInProgressEvent, ResponseCompletedEvent]

    @pytest.mark.anyio
    async def test_tool_message_update_emits_server_function_call(self):
        from langchain_core.messages import AIMessageChunk, ToolMessage

        graph = _FakePartsGraph(
            [
                {
                    "type": "messages",
                    "data": (
                        AIMessageChunk(
                            content="",
                            tool_call_chunks=[
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "name": "lookup",
                                    "args": '{"query": "x"}',
                                }
                            ],
                        ),
                        {},
                    ),
                },
                {
                    "type": "updates",
                    "data": {
                        "agent": {
                            "messages": [
                                ToolMessage(content="result", tool_call_id="call_1")
                            ]
                        }
                    },
                },
            ]
        )
        backend = LangchainBackend(graph)
        ctx = make_ctx()

        events = [e async for e in backend.handle(ctx)]
        server_done = [
            event.item
            for event in events
            if isinstance(event, ResponseOutputItemDoneEvent)
            and isinstance(event.item, ServerFunctionCall)
        ]

        assert len(server_done) == 1
        assert server_done[0].name == "lookup"
        assert server_done[0].output == "result"
        assert isinstance(events[-1], ResponseCompletedEvent)
        assert len(events[-1].response.output) == 1
