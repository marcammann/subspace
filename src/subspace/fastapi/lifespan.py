from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI


def combine_lifespans(*contexts: Any) -> Callable[[FastAPI], Any]:
    """Combine multiple async context managers into a single FastAPI lifespan.

    Each context can be a Subspace instance, an async context manager, or any object
    implementing __aenter__/__aexit__. They are entered in order and exited in reverse.

        app = FastAPI(lifespan=combine_lifespans(subspace_a, subspace_b))
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, Any]]:
        async with AsyncExitStack() as stack:
            for ctx in contexts:
                await stack.enter_async_context(ctx)
            yield {}

    return lifespan
