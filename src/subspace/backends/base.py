from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from subspace.middleware.context import RequestContext
from subspace.models.agent import AgentCapabilities
from subspace.models.events import StreamEvent


class Backend(Protocol):
    def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]: ...


@runtime_checkable
class CapabilityProvider(Protocol):
    @property
    def capabilities(self) -> AgentCapabilities:
        """Capabilities provided by this backend before middleware transforms."""
        ...
