
import pytest
from conftest import StaticBackend, error_backend_events, text_backend_events
from httpx import ASGITransport, AsyncClient

from subspace.fastapi import OpenResponsesRouter, SubspaceApp, SubspaceMount


def _make_app() -> SubspaceApp:
    mount = SubspaceMount(interfaces=[OpenResponsesRouter(prefix="/v1")])
    mount.agent("echo", backend=StaticBackend(text_backend_events()))
    return SubspaceApp(mount)


def _make_error_app() -> SubspaceApp:
    mount = SubspaceMount(interfaces=[OpenResponsesRouter(prefix="/v1")])
    mount.agent("echo", backend=StaticBackend(error_backend_events()))
    return SubspaceApp(mount)


@pytest.fixture
def app():
    return _make_app()


class TestPostResponses:
    @pytest.mark.anyio
    async def test_stream_response(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/responses",
                json={"model": "echo", "input": "hello", "stream": True},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = resp.text.strip().split("\n")
            assert any("[DONE]" in line for line in lines)

    @pytest.mark.anyio
    async def test_non_stream_response(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/responses",
                json={"model": "echo", "input": "hello", "stream": False},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "completed"
            assert len(body["output"]) > 0

    @pytest.mark.anyio
    async def test_unknown_model_404(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/responses",
                json={"model": "nonexistent", "input": "hello", "stream": False},
            )
            assert resp.status_code == 404
            assert "model_not_found" in resp.json()["error"]["code"]

    @pytest.mark.anyio
    async def test_non_stream_failed_response_is_returned(self):
        async with AsyncClient(
            transport=ASGITransport(app=_make_error_app()), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/responses",
                json={"model": "echo", "input": "hello", "stream": False},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "failed"


class TestGetModels:
    @pytest.mark.anyio
    async def test_list_models(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            model_ids = [m["id"] for m in data["data"]]
            assert "echo" in model_ids
