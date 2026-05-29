from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack

from subspace.backends.base import Backend, CapabilityProvider
from subspace.middleware.base import Middleware
from subspace.middleware.chain import MiddlewareChain
from subspace.models.agent import AgentCapabilities, AgentCard, AgentRuntime, Skill


class _RegisteredAgent:
    __slots__ = ("backend", "middlewares", "card")

    def __init__(
        self, backend: Backend, middlewares: list[Middleware], card: AgentCard
    ) -> None:
        self.backend = backend
        self.middlewares = middlewares
        self.card = card


class Agent:
    """Handle to a registered agent. Carries metadata and can build its chain."""

    def __init__(self, name: str, subspace: "Subspace", card: AgentCard) -> None:
        self._name = name
        self._subspace = subspace
        self.card = card

    @property
    def name(self) -> str:
        return self._name

    def build_chain(
        self, extra_middlewares: Sequence[Middleware] = ()
    ) -> MiddlewareChain:
        return self._subspace.build_chain(self._name, extra_middlewares)

    def capabilities(
        self, extra_middlewares: Sequence[Middleware] = ()
    ) -> AgentCapabilities:
        """Return capabilities for this agent with optional interface middleware."""
        return self._subspace.capabilities(self._name, extra_middlewares)


class Subspace:
    def __init__(self) -> None:
        self._agents: dict[str, _RegisteredAgent] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self):
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        for backend in self.all_backends():
            if is_async_context_manager(backend):
                await self._exit_stack.enter_async_context(backend)
        for mw in self.all_middlewares():
            await self._exit_stack.enter_async_context(mw)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None

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
        agent_middlewares = list(middlewares)
        capabilities = _resolve_capabilities(backend, agent_middlewares)
        card = AgentCard(
            name=name,
            description=description,
            skills=list(skills),
            runtime=runtime,
            capabilities=capabilities.model_copy(
                update={"provisioned_runtime": runtime is AgentRuntime.PROVISIONED}
            ),
        )
        self._agents[name] = _RegisteredAgent(
            backend=backend,
            middlewares=agent_middlewares,
            card=card,
        )
        return Agent(name=name, subspace=self, card=card)

    def resolve_agent(self, name: str) -> _RegisteredAgent | None:
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    def build_chain(
        self,
        agent_name: str,
        interface_middlewares: Sequence[Middleware] = (),
    ) -> MiddlewareChain:
        registered = self._agents.get(agent_name)
        if registered is None:
            raise AgentNotFoundError(agent_name)

        combined = list(interface_middlewares) + registered.middlewares
        return MiddlewareChain(middlewares=combined, backend=registered.backend)

    def capabilities(
        self,
        agent_name: str,
        interface_middlewares: Sequence[Middleware] = (),
    ) -> AgentCapabilities:
        """Return effective capabilities for an agent and interface middleware."""
        registered = self._agents.get(agent_name)
        if registered is None:
            raise AgentNotFoundError(agent_name)
        return _resolve_capabilities(
            registered.backend,
            [*interface_middlewares, *registered.middlewares],
            base=registered.card.capabilities,
        )

    def all_middlewares(self) -> list[Middleware]:
        seen: set[int] = set()
        result: list[Middleware] = []
        for reg in self._agents.values():
            for mw in reg.middlewares:
                if id(mw) not in seen:
                    seen.add(id(mw))
                    result.append(mw)
        return result

    def all_backends(self) -> list[Backend]:
        """Return unique registered backend instances."""
        seen: set[int] = set()
        result: list[Backend] = []
        for reg in self._agents.values():
            if id(reg.backend) not in seen:
                seen.add(id(reg.backend))
                result.append(reg.backend)
        return result


class AgentNotFoundError(Exception):
    def __init__(self, agent: str) -> None:
        self.agent = agent
        super().__init__(f"Agent not found: {agent}")


def _resolve_capabilities(
    backend: Backend,
    middlewares: Sequence[Middleware],
    *,
    base: AgentCapabilities | None = None,
) -> AgentCapabilities:
    capabilities = base
    if capabilities is None:
        if isinstance(backend, CapabilityProvider):
            capabilities = backend.capabilities
        else:
            capabilities = AgentCapabilities()

    for middleware in middlewares:
        capabilities = middleware.transform_capabilities(capabilities)
    return capabilities


def is_async_context_manager(obj: object) -> bool:
    """Return whether an object provides async context manager lifecycle hooks."""
    return isinstance(obj, AbstractAsyncContextManager)
