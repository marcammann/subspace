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
from subspace.fastapi.routers.chatcompletions import ChatCompletionsRouter


def _make_app(backend=None) -> SubspaceApp:
    mount = SubspaceMount(interfaces=[ChatCompletionsRouter(prefix="/v1")])
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
                "/v1/chat/completions",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = resp.text.strip().split("\n")
            data_lines = [line for line in lines if line.startswith("data:")]
            assert any("[DONE]" in line for line in data_lines)
            assert not any(line.startswith("event:") for line in lines)

    @pytest.mark.anyio
    async def test_stream_contains_content_deltas(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
            data_lines = [
                line.removeprefix("data:").strip()
                for line in resp.text.split("\n")
                if line.startswith("data:") and "[DONE]" not in line
            ]
            chunks = [json.loads(d) for d in data_lines if d]
            assert any(c["choices"][0]["delta"].get("content") for c in chunks)
            assert all(c["object"] == "chat.completion.chunk" for c in chunks)

    @pytest.mark.anyio
    async def test_stream_tool_calls(self, tool_app):
        async with AsyncClient(
            transport=ASGITransport(app=tool_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "stream": True,
                },
            )
            data_lines = [
                line.removeprefix("data:").strip()
                for line in resp.text.split("\n")
                if line.startswith("data:") and "[DONE]" not in line
            ]
            chunks = [json.loads(d) for d in data_lines if d]
            tool_chunks = [
                c for c in chunks if c["choices"][0]["delta"].get("tool_calls")
            ]
            assert len(tool_chunks) > 0
            final = [c for c in chunks if c["choices"][0].get("finish_reason")]
            assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"


class TestNonStreamResponse:
    @pytest.mark.anyio
    async def test_non_stream_returns_json(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["object"] == "chat.completion"
            assert body["choices"][0]["message"]["role"] == "assistant"
            assert body["choices"][0]["message"]["content"] is not None
            assert body["choices"][0]["finish_reason"] == "stop"
            assert "usage" in body

    @pytest.mark.anyio
    async def test_non_stream_tool_calls(self, tool_app):
        async with AsyncClient(
            transport=ASGITransport(app=tool_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "stream": False,
                },
            )
            body = resp.json()
            assert body["choices"][0]["finish_reason"] == "tool_calls"
            tool_calls = body["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["function"]["name"] == "get_weather"


class TestErrors:
    @pytest.mark.anyio
    async def test_unknown_model_404(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "nonexistent",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 404
            assert "model_not_found" in resp.json()["error"]["code"]

    @pytest.mark.anyio
    async def test_failed_backend_returns_wire_error(self):
        app = _make_app(backend=StaticBackend(error_backend_events()))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "echo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 502
            assert resp.json()["error"]["code"] == "server_error"


class TestInboundTranslation:
    @pytest.mark.anyio
    async def test_system_message_extracted(self):
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
                "/v1/chat/completions",
                json={
                    "model": "echo",
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "hello"},
                    ],
                    "stream": False,
                },
            )
        assert len(captured_instructions) == 1
        assert captured_instructions[0] == "You are helpful"
