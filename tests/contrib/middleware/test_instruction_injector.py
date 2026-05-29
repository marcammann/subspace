from collections.abc import AsyncIterator

import pytest
from conftest import make_ctx, text_backend_events

from subspace.contrib.middleware.instruction_injector import InstructionInjectorMiddleware
from subspace.middleware.context import RequestContext
from subspace.models.events import StreamEvent


async def _call_next_with(events: list[StreamEvent]):
    async def call_next(ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        for e in events:
            yield e

    return call_next


class TestInstructionInjector:
    @pytest.mark.anyio
    async def test_template_substitution(self):
        mw = InstructionInjectorMiddleware("Be helpful.\n\n{instructions}\n\nBe concise.")
        ctx = make_ctx(instructions="User rules.")
        call_next = await _call_next_with(text_backend_events())

        _ = [e async for e in mw(ctx, call_next)]
        assert ctx.request.instructions == "Be helpful.\n\nUser rules.\n\nBe concise."

    @pytest.mark.anyio
    async def test_appends_without_placeholder(self):
        mw = InstructionInjectorMiddleware("Be helpful.")
        ctx = make_ctx(instructions="User rules.")
        call_next = await _call_next_with(text_backend_events())

        _ = [e async for e in mw(ctx, call_next)]
        assert ctx.request.instructions == "User rules.\n\nBe helpful."

    @pytest.mark.anyio
    async def test_no_existing_instructions(self):
        mw = InstructionInjectorMiddleware("Be helpful.")
        ctx = make_ctx()
        call_next = await _call_next_with(text_backend_events())

        _ = [e async for e in mw(ctx, call_next)]
        assert ctx.request.instructions == "Be helpful."

    @pytest.mark.anyio
    async def test_template_no_existing_instructions(self):
        mw = InstructionInjectorMiddleware("Persona.\n\n{instructions}\n\nFormat: JSON.")
        ctx = make_ctx()
        call_next = await _call_next_with(text_backend_events())

        _ = [e async for e in mw(ctx, call_next)]
        assert ctx.request.instructions == "Persona.\n\n\n\nFormat: JSON."

    @pytest.mark.anyio
    async def test_events_pass_through(self):
        mw = InstructionInjectorMiddleware("Be helpful.")
        ctx = make_ctx()
        backend_events = text_backend_events()
        call_next = await _call_next_with(backend_events)

        events = [e async for e in mw(ctx, call_next)]
        assert len(events) == len(backend_events)

    @pytest.mark.anyio
    async def test_nested_layers_compose(self):
        """Outer template wraps inner, inner wraps original."""
        outer = InstructionInjectorMiddleware("Outer.\n\n{instructions}")
        inner = InstructionInjectorMiddleware("Inner.\n\n{instructions}")
        ctx = make_ctx(instructions="Original.")
        events = text_backend_events()

        async def backend(ctx: RequestContext) -> AsyncIterator[StreamEvent]:
            for e in events:
                yield e

        # Simulate middleware ordering: outer runs first, inner runs second
        async def inner_call_next(ctx: RequestContext) -> AsyncIterator[StreamEvent]:
            async for e in inner(ctx, backend):
                yield e

        _ = [e async for e in outer(ctx, inner_call_next)]
        assert ctx.request.instructions == "Outer.\n\nInner.\n\nOriginal."
