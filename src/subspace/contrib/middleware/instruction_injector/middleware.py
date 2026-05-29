from collections.abc import AsyncIterator

from subspace.middleware.base import NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.events import StreamEvent


class InstructionInjectorMiddleware:
    """Injects instructions into the request via a template.

    If the template contains ``{instructions}``, the existing request
    instructions are substituted at that position.  Otherwise the template
    text is appended after any existing instructions.

    Multiple injectors compose correctly: outer middleware templates wrap
    inner ones because the resolution is deferred until ``ctx.request``
    is read.
    """

    def __init__(self, instructions: str) -> None:
        self._instructions = instructions

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        ctx.add_instructions(self._instructions)

        async for event in call_next(ctx):
            yield event
