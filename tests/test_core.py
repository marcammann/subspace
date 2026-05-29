import pytest
from conftest import StaticBackend, text_backend_events

from subspace.core import Agent, AgentNotFoundError, Subspace
from subspace.middleware.base import Middleware
from subspace.middleware.chain import MiddlewareChain
from subspace.models.agent import AgentCapabilities


class TestSubspace:
    def test_register_and_resolve(self):
        s = Subspace()
        backend = StaticBackend(text_backend_events())
        s.agent("echo", backend=backend)
        reg = s.resolve_agent("echo")
        assert reg is not None
        assert reg.backend is backend
        assert reg.middlewares == []

    def test_resolve_unknown_returns_none(self):
        s = Subspace()
        assert s.resolve_agent("nope") is None

    def test_list_agents(self):
        s = Subspace()
        backend = StaticBackend(text_backend_events())
        s.agent("a", backend=backend)
        s.agent("b", backend=backend)
        assert sorted(s.list_agents()) == ["a", "b"]

    def test_build_chain_returns_chain(self):
        s = Subspace()
        s.agent("echo", backend=StaticBackend(text_backend_events()))
        chain = s.build_chain("echo")
        assert isinstance(chain, MiddlewareChain)

    def test_build_chain_unknown_raises(self):
        s = Subspace()
        with pytest.raises(AgentNotFoundError):
            s.build_chain("nope")

    def test_build_chain_combines_middlewares(self):
        s = Subspace()
        agent_mw = Middleware()
        interface_mw = Middleware()
        s.agent("echo", backend=StaticBackend(text_backend_events()), middlewares=[agent_mw])
        chain = s.build_chain("echo", interface_middlewares=[interface_mw])
        assert chain._middlewares == [interface_mw, agent_mw]

    @pytest.mark.anyio
    async def test_context_manager_lifecycle(self):
        entered = []
        exited = []

        class LifecycleMw(Middleware):
            async def __aenter__(self):
                entered.append(True)
                return self

            async def __aexit__(self, *args):
                exited.append(True)

        mw = LifecycleMw()
        s = Subspace()
        s.agent("echo", backend=StaticBackend(text_backend_events()), middlewares=[mw])
        async with s:
            assert len(entered) == 1
        assert len(exited) == 1

    @pytest.mark.anyio
    async def test_context_manager_enters_backend_lifecycle(self):
        entered = []
        exited = []

        class LifecycleBackend(StaticBackend):
            async def __aenter__(self):
                entered.append(True)
                return self

            async def __aexit__(self, *args):
                exited.append(True)

        s = Subspace()
        s.agent("echo", backend=LifecycleBackend(text_backend_events()))
        async with s:
            assert len(entered) == 1
        assert len(exited) == 1

    def test_all_middlewares_deduplicates(self):
        s = Subspace()
        mw = Middleware()
        backend = StaticBackend(text_backend_events())
        s.agent("a", backend=backend, middlewares=[mw])
        s.agent("b", backend=backend, middlewares=[mw])
        assert len(s.all_middlewares()) == 1

    def test_agent_returns_handle(self):
        s = Subspace()
        handle = s.agent("echo", backend=StaticBackend(text_backend_events()), description="test")
        assert isinstance(handle, Agent)
        assert handle.name == "echo"
        assert handle.card.description == "test"

    def test_agent_handle_build_chain(self):
        s = Subspace()
        handle = s.agent("echo", backend=StaticBackend(text_backend_events()))
        chain = handle.build_chain()
        assert isinstance(chain, MiddlewareChain)

    def test_capabilities_apply_middleware_transforms(self):
        class ToolMiddleware(Middleware):
            def transform_capabilities(self, capabilities: AgentCapabilities) -> AgentCapabilities:
                return capabilities.model_copy(update={"function_tools": True})

        s = Subspace()
        handle = s.agent(
            "echo",
            backend=StaticBackend(text_backend_events()),
            middlewares=[ToolMiddleware()],
        )

        assert handle.card.capabilities.function_tools is True
        assert handle.capabilities().function_tools is True
