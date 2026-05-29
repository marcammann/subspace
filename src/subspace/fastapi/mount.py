from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from fastapi import APIRouter

from subspace.backends.base import Backend
from subspace.core import Agent, Subspace
from subspace.middleware.base import Middleware
from subspace.middleware.context import RequestContext
from subspace.models.agent import AgentRuntime, Skill


class Interface(Protocol):
    def build_router(self, mount: "SubspaceMount") -> APIRouter: ...


class SubspaceMount:
    """Binds a Subspace registry to interfaces with shared configuration.

        mount = SubspaceMount(
            deps=get_deps,
            context_class=MyContext,
            interfaces=[OpenResponsesRouter(prefix="/v1")],
        )
        weather = mount.agent("weather", backend=litellm, middlewares=[mcp])

        app = SubspaceApp(mount)
    """

    def __init__(
        self,
        subspace: Subspace | None = None,
        *,
        interfaces: Sequence[Interface] = (),
        middlewares: Sequence[Middleware] = (),
        lifespan: Sequence[AbstractAsyncContextManager[Any]] = (),
        deps: Callable[..., Any] | None = None,
        context_class: type[RequestContext] = RequestContext,
    ) -> None:
        self._subspace = subspace or Subspace()
        self._interfaces = list(interfaces)
        self.lifespan = list(lifespan)
        self.middlewares = list(middlewares)
        self.deps = deps
        self.context_class = context_class

    @property
    def subspace(self) -> Subspace:
        return self._subspace

    @property
    def interfaces(self) -> list[Interface]:
        return self._interfaces

    def agent(
        self,
        name: str,
        *,
        backend: Backend,
        middlewares: Sequence[Middleware] = (),
        description: str = "",
        skills: Sequence[Skill] = (),
        runtime: AgentRuntime = AgentRuntime.INLINE,
    ) -> Agent:
        return self._subspace.agent(
            name, backend=backend, middlewares=middlewares,
            description=description, skills=skills,
            runtime=runtime,
        )
