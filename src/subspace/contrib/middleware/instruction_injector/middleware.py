from collections.abc import AsyncIterator

from subspace.middleware.base import NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.events import StreamEvent


class InstructionInjectorMiddleware:
    """Merges static system instructions into the request.

    Prepends or appends the configured instructions to whatever instructions
    the request already carries. Useful for injecting persona, constraints, or
    formatting rules at the model or interface level.
    """

    def __init__(self, instructions: str, position: str = "prepend") -> None:
        self._instructions = instructions
        self._position = position

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        existing = ctx.request.instructions or ""
        if self._position == "prepend":
            merged = f"{self._instructions}\n\n{existing}".strip()
        else:
            merged = f"{existing}\n\n{self._instructions}".strip()

        ctx.request = ctx.request.model_copy(update={"instructions": merged})

        async for event in call_next(ctx):
            yield event
