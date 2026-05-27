from collections.abc import Sequence
from contextlib import AsyncExitStack

from subspace.backends.base import Backend
from subspace.middleware.base import Middleware
from subspace.middleware.chain import MiddlewareChain


class _RegisteredModel:
    __slots__ = ("backend", "middlewares")

    def __init__(self, backend: Backend, middlewares: list[Middleware]) -> None:
        self.backend = backend
        self.middlewares = middlewares


class Subspace:
    def __init__(self) -> None:
        self._models: dict[str, _RegisteredModel] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self):
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        for mw in self.all_middlewares():
            await self._exit_stack.enter_async_context(mw)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None

    def model(
        self,
        name: str,
        *,
        backend: Backend,
        middlewares: Sequence[Middleware] = (),
    ) -> None:
        self._models[name] = _RegisteredModel(
            backend=backend,
            middlewares=list(middlewares),
        )

    def resolve(self, name: str) -> _RegisteredModel | None:
        return self._models.get(name)

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def build_chain(
        self,
        model_name: str,
        interface_middlewares: Sequence[Middleware] = (),
    ) -> MiddlewareChain:
        registered = self._models.get(model_name)
        if registered is None:
            raise ModelNotFoundError(model_name)

        combined = list(interface_middlewares) + registered.middlewares
        return MiddlewareChain(middlewares=combined, backend=registered.backend)

    def all_middlewares(self) -> list[Middleware]:
        seen: set[int] = set()
        result: list[Middleware] = []
        for reg in self._models.values():
            for mw in reg.middlewares:
                if id(mw) not in seen:
                    seen.add(id(mw))
                    result.append(mw)
        return result


class ModelNotFoundError(Exception):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model not found: {model}")
