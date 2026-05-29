from collections.abc import AsyncIterator

import pytest
from conftest import text_backend_events

from subspace.middleware.context import RequestContext
from subspace.middleware.conversation_history import ConversationHistoryMiddleware, InMemoryStorage
from subspace.models.common import Status
from subspace.models.content import OutputTextContent
from subspace.models.events import StreamEvent
from subspace.models.items import InputMessage, OutputMessage
from subspace.models.request import CreateResponseRequest
from subspace.models.response import ResponseResource


def _make_history_ctx(
    input_text: str = "hello",
    previous_response_id: str | None = None,
) -> RequestContext:
    request = CreateResponseRequest(
        model="test",
        input=input_text,
        previous_response_id=previous_response_id,
    )
    return RequestContext(
        request=request,
        response_id="resp_new",
        response=ResponseResource(
            id="resp_new", created_at=0, status=Status.IN_PROGRESS, model="test"
        ),
    )


async def _call_next_with(events: list[StreamEvent]):
    async def call_next(ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        for e in events:
            yield e

    return call_next


class TestInMemoryStorage:
    @pytest.mark.anyio
    async def test_save_and_load_roundtrip(self):
        storage = InMemoryStorage()
        input_items = [InputMessage(role="user", content="hi")]
        output_items = [
            OutputMessage(
                id="msg_1", content=[OutputTextContent(text="hello")], status=Status.COMPLETED
            )
        ]
        await storage.save_response("resp_1", input_items, output_items)
        history = await storage.load_history("resp_1")
        assert len(history) == 2

    @pytest.mark.anyio
    async def test_load_missing_returns_empty(self):
        storage = InMemoryStorage()
        assert await storage.load_history("nonexistent") == []


class TestConversationHistoryMiddleware:
    @pytest.mark.anyio
    async def test_no_history_without_previous_id(self):
        storage = InMemoryStorage()
        mw = ConversationHistoryMiddleware(storage)
        ctx = _make_history_ctx("hello")
        call_next = await _call_next_with(text_backend_events())

        original_input = list(ctx.request.input)
        events = [e async for e in mw(ctx, call_next)]
        assert ctx.request.input == original_input
        assert len(events) > 0

    @pytest.mark.anyio
    async def test_saves_on_completed(self):
        storage = InMemoryStorage()
        mw = ConversationHistoryMiddleware(storage)
        ctx = _make_history_ctx("hello")
        call_next = await _call_next_with(text_backend_events())

        _ = [e async for e in mw(ctx, call_next)]
        history = await storage.load_history("resp_new")
        assert len(history) > 0

    @pytest.mark.anyio
    async def test_prepends_history(self):
        storage = InMemoryStorage()
        input_items = [InputMessage(role="user", content="first")]
        output_items = [
            OutputMessage(
                id="msg_1", content=[OutputTextContent(text="response")], status=Status.COMPLETED
            )
        ]
        await storage.save_response("resp_prev", input_items, output_items)

        mw = ConversationHistoryMiddleware(storage)
        ctx = _make_history_ctx("second", previous_response_id="resp_prev")
        call_next = await _call_next_with(text_backend_events())

        _ = [e async for e in mw(ctx, call_next)]
        assert len(ctx.request.input) == 3
