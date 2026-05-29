from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI

from subspace.core import is_async_context_manager
from subspace.fastapi.mount import SubspaceMount


class SubspaceApp(FastAPI):
    """FastAPI application that manages SubspaceMount lifecycles.

        mount = SubspaceMount(
            interfaces=[OpenResponsesRouter(prefix="/v1")],
        )
        mount.agent("claude", backend=litellm, middlewares=[mcp])

        app = SubspaceApp(mount)
    """

    def __init__(self, *mounts: SubspaceMount, **fastapi_kwargs: Any) -> None:
        self._mounts = list(mounts)
        fastapi_kwargs.setdefault("lifespan", self._lifespan)
        super().__init__(**fastapi_kwargs)

        for mount in self._mounts:
            for interface in mount.interfaces:
                self.include_router(interface.build_router(mount))

    @staticmethod
    @asynccontextmanager
    async def _lifespan(app: "SubspaceApp") -> AsyncIterator[dict[str, Any]]:
        async with AsyncExitStack() as stack:
            seen: set[int] = set()
            seen_middlewares: set[int] = set()
            for mount in app._mounts:
                if id(mount.subspace) not in seen:
                    seen.add(id(mount.subspace))
                    await stack.enter_async_context(mount.subspace)
                for middleware in mount.middlewares:
                    if id(middleware) in seen_middlewares:
                        continue
                    seen_middlewares.add(id(middleware))
                    if is_async_context_manager(middleware):
                        await stack.enter_async_context(middleware)
                for ctx in mount.lifespan:
                    await stack.enter_async_context(ctx)
            yield {}
