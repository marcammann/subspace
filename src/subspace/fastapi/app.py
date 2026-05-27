from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI

from subspace.fastapi.mount import SubspaceMount


class SubspaceApp(FastAPI):
    """FastAPI application that manages SubspaceMount lifecycles.

        mount = SubspaceMount(
            interfaces=[OpenResponsesInterface(prefix="/v1")],
        )
        mount.model("claude", backend=litellm, middlewares=[mcp])

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
            for mount in app._mounts:
                if id(mount.subspace) not in seen:
                    seen.add(id(mount.subspace))
                    await stack.enter_async_context(mount.subspace)
                for ctx in mount.lifespan:
                    await stack.enter_async_context(ctx)
            yield {}
