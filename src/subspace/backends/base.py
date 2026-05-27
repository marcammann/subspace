from collections.abc import AsyncIterator
from typing import Protocol

from subspace.middleware.context import RequestContext
from subspace.models.events import StreamEvent


class Backend(Protocol):
    def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]: ...
