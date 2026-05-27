from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from subspace.backends.base import Backend
from subspace.core import Subspace
from subspace.middleware.base import Middleware
from subspace.middleware.context import RequestContext

if TYPE_CHECKING:
    from subspace.fastapi.openresponses import OpenResponsesInterface


class SubspaceMount:
    """Binds a Subspace registry to interfaces with shared configuration.

        mount = SubspaceMount(
            deps=get_deps,
            context_class=MyContext,
            interfaces=[OpenResponsesInterface(prefix="/v1")],
        )
        mount.model("claude", backend=litellm, middlewares=[mcp])

        app = SubspaceApp(mount)
    """

    def __init__(
        self,
        subspace: Subspace | None = None,
        *,
        interfaces: Sequence["OpenResponsesInterface"] = (),
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
    def interfaces(self) -> list["OpenResponsesInterface"]:
        return self._interfaces

    def model(
        self,
        name: str,
        *,
        backend: Backend,
        middlewares: Sequence[Middleware] = (),
    ) -> None:
        self._subspace.model(name, backend=backend, middlewares=middlewares)
