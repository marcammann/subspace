import json

import pytest
from conftest import (
    StaticBackend,
    error_backend_events,
    function_call_backend_events,
    text_backend_events,
)
from httpx import ASGITransport, AsyncClient

from subspace.fastapi import SubspaceApp, SubspaceMount
from subspace.fastapi.routers.anthropic import AnthropicMessagesRouter


def _make_app(backend=None) -> SubspaceApp:
    mount = SubspaceMount(interfaces=[AnthropicMessagesRouter(prefix="/v1")])
    mount.agent("echo", backend=backend or StaticBackend(text_backend_events()))
    return SubspaceApp(mount)


@pytest.fixture
def app():
    return _make_app()


@pytest.fixture
def tool_app():
    return _make_app(backend=StaticBackend(function_call_backend_events()))


class TestStreamResponse:
    @pytest.mark.anyio
    async def test_stream_returns_sse(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1024,
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = resp.text.strip().split("\n")
            event_lines = [
                line.removeprefix("event:").strip()
                for line in lines
                if line.startswith("event:")
            ]
            assert "message_start" in event_lines
            assert "content_block_start" in event_lines
            assert "content_block_delta" in event_lines
            assert "content_block_stop" in event_lines
            assert "message_delta" in event_lines
            assert "message_stop" in event_lines

    @pytest.mark.anyio
    async def test_stream_text_deltas(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1024,
                    "stream": True,
                },
            )
            data_lines = [
                line.removeprefix("data:").strip()
                for line in resp.text.split("\n")
                if line.startswith("data:")
            ]
            events = [json.loads(d) for d in data_lines if d]
            text_deltas = [
                e for e in events
                if e.get("type") == "content_block_delta"
                and e.get("delta", {}).get("type") == "text_delta"
            ]
            assert len(text_deltas) > 0
            combined = "".join(d["delta"]["text"] for d in text_deltas)
            assert "Hello" in combined

    @pytest.mark.anyio
    async def test_stream_tool_use(self, tool_app):
        async with AsyncClient(
            transport=ASGITransport(app=tool_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "max_tokens": 1024,
                    "stream": True,
                },
            )
            data_lines = [
                line.removeprefix("data:").strip()
                for line in resp.text.split("\n")
                if line.startswith("data:")
            ]
            events = [json.loads(d) for d in data_lines if d]
            tool_starts = [
                e for e in events
                if e.get("type") == "content_block_start"
                and e.get("content_block", {}).get("type") == "tool_use"
            ]
            assert len(tool_starts) == 1
            assert tool_starts[0]["content_block"]["name"] == "get_weather"

            json_deltas = [
                e for e in events
                if e.get("type") == "content_block_delta"
                and e.get("delta", {}).get("type") == "input_json_delta"
            ]
            assert len(json_deltas) > 0

            msg_delta = [e for e in events if e.get("type") == "message_delta"]
            assert msg_delta[-1]["delta"]["stop_reason"] == "tool_use"


class TestNonStreamResponse:
    @pytest.mark.anyio
    async def test_non_stream_returns_json(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1024,
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["type"] == "message"
            assert body["role"] == "assistant"
            assert body["stop_reason"] == "end_turn"
            assert len(body["content"]) > 0
            assert body["content"][0]["type"] == "text"
            assert "Hello" in body["content"][0]["text"]
            assert "usage" in body

    @pytest.mark.anyio
    async def test_non_stream_tool_use(self, tool_app):
        async with AsyncClient(
            transport=ASGITransport(app=tool_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "max_tokens": 1024,
                    "stream": False,
                },
            )
            body = resp.json()
            assert body["stop_reason"] == "tool_use"
            tool_blocks = [c for c in body["content"] if c["type"] == "tool_use"]
            assert len(tool_blocks) == 1
            assert tool_blocks[0]["name"] == "get_weather"
            assert isinstance(tool_blocks[0]["input"], dict)


class TestErrors:
    @pytest.mark.anyio
    async def test_unknown_model_404(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "nonexistent",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1024,
                    "stream": False,
                },
            )
            assert resp.status_code == 404
            body = resp.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "not_found_error"

    @pytest.mark.anyio
    async def test_failed_backend_returns_wire_error(self):
        app = _make_app(backend=StaticBackend(error_backend_events()))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1024,
                    "stream": False,
                },
            )
            assert resp.status_code == 502
            assert resp.json()["error"]["type"] == "server_error"


class TestInboundTranslation:
    @pytest.mark.anyio
    async def test_system_string_extracted(self):
        from collections.abc import AsyncIterator

        from subspace.middleware.context import RequestContext
        from subspace.models.events import StreamEvent

        captured_instructions = []

        class CapturingBackend:
            async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
                captured_instructions.append(ctx.request.instructions)
                for event in text_backend_events():
                    yield event

        app = _make_app(backend=CapturingBackend())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1024,
                    "system": "You are helpful",
                },
            )
        assert len(captured_instructions) == 1
        assert captured_instructions[0] == "You are helpful"

    @pytest.mark.anyio
    async def test_system_list_extracted(self):
        from collections.abc import AsyncIterator

        from subspace.middleware.context import RequestContext
        from subspace.models.events import StreamEvent

        captured_instructions = []

        class CapturingBackend:
            async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
                captured_instructions.append(ctx.request.instructions)
                for event in text_backend_events():
                    yield event

        app = _make_app(backend=CapturingBackend())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/messages",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1024,
                    "system": [{"type": "text", "text": "Be helpful"}],
                },
            )
        assert captured_instructions[0] == "Be helpful"
