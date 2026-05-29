import logging

import pytest
from conftest import StaticBackend, make_ctx, text_backend_events

from subspace.contrib.middleware.logging import LoggingMiddleware
from subspace.middleware.chain import MiddlewareChain


class TestLoggingMiddleware:
    @pytest.mark.anyio
    async def test_all_events_yielded(self):
        backend_events = text_backend_events()
        chain = MiddlewareChain(
            middlewares=[LoggingMiddleware()],
            backend=StaticBackend(backend_events),
        )
        events = [e async for e in chain.execute(make_ctx())]
        assert len(events) == len(backend_events)

    @pytest.mark.anyio
    async def test_logs_emitted(self, caplog):
        chain = MiddlewareChain(
            middlewares=[LoggingMiddleware()],
            backend=StaticBackend(text_backend_events()),
        )
        with caplog.at_level(logging.INFO, logger="subspace"):
            _ = [e async for e in chain.execute(make_ctx())]
        assert len(caplog.records) > 0
