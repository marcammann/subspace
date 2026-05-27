"""Langfuse prompt middleware — fetches prompts from Langfuse and injects them into requests.

Supports both text prompts (injected as instructions) and chat prompts (injected
as input messages). Prompts are cached by the Langfuse SDK (default 60s TTL).

Prompt variables can be resolved statically at construction time, or dynamically
per-request via a callback.

Usage:
    from subspace.contrib.middleware.langfuse_prompt import LangfusePromptMiddleware

    # Text prompt as system instructions
    mw = LangfusePromptMiddleware(
        prompt_name="assistant-persona",
        label="production",
    )

    # Chat prompt with variables
    mw = LangfusePromptMiddleware(
        prompt_name="customer-support",
        prompt_type="chat",
        variables={"company": "Acme Corp"},
    )

    # Dynamic variables resolved per-request
    mw = LangfusePromptMiddleware(
        prompt_name="personalized-assistant",
        variables=lambda ctx: {"user_name": ctx.metadata.get("user_name", "there")},
    )

    # With a custom Langfuse client
    from langfuse import Langfuse
    lf = Langfuse(public_key="pk-...", secret_key="sk-...", host="https://langfuse.example.com")
    mw = LangfusePromptMiddleware(prompt_name="my-prompt", langfuse=lf)
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from subspace.middleware.base import NextHandler
from subspace.middleware.context import RequestContext
from subspace.models.common import Role
from subspace.models.events import StreamEvent
from subspace.models.items import InputMessage

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.model import ChatPromptClient, PromptClient, TextPromptClient


class LangfusePromptMiddleware:
    """Fetches a prompt from Langfuse and injects it into the request.

    Text prompts are merged into the request's instructions field.
    Chat prompts are prepended to the request's input as messages.

    The prompt's config dict (set in the Langfuse UI) is merged into
    ctx.state["langfuse_prompt_config"] for downstream middlewares
    or backends to use (e.g. for model parameters, temperature overrides).
    """

    def __init__(
        self,
        prompt_name: str,
        *,
        prompt_type: str = "text",
        version: int | None = None,
        label: str | None = None,
        variables: dict[str, str] | Callable[[RequestContext], dict[str, str]] | None = None,
        position: str = "prepend",
        cache_ttl_seconds: int | None = None,
        fallback: str | list[dict[str, str]] | None = None,
        langfuse: "Langfuse | None" = None,
    ) -> None:
        self._prompt_name = prompt_name
        self._prompt_type = prompt_type
        self._version = version
        self._label = label
        self._variables = variables
        self._position = position
        self._cache_ttl_seconds = cache_ttl_seconds
        self._fallback = fallback
        self._langfuse = langfuse

    def _get_langfuse(self) -> "Langfuse":
        if self._langfuse is None:
            from langfuse import Langfuse

            self._langfuse = Langfuse()
        return self._langfuse

    def _resolve_variables(self, ctx: RequestContext) -> dict[str, str]:
        if self._variables is None:
            return {}
        if callable(self._variables):
            return self._variables(ctx)
        return self._variables

    def _fetch_prompt(self, ctx: RequestContext) -> "PromptClient":
        langfuse = self._get_langfuse()
        kwargs: dict[str, Any] = {
            "name": self._prompt_name,
            "type": self._prompt_type,
        }
        if self._version is not None:
            kwargs["version"] = self._version
        if self._label is not None:
            kwargs["label"] = self._label
        if self._cache_ttl_seconds is not None:
            kwargs["cache_ttl_seconds"] = self._cache_ttl_seconds
        if self._fallback is not None:
            kwargs["fallback"] = self._fallback
        return langfuse.get_prompt(**kwargs)

    async def __call__(
        self,
        ctx: RequestContext,
        call_next: NextHandler,
    ) -> AsyncIterator[StreamEvent]:
        prompt = await asyncio.to_thread(self._fetch_prompt, ctx)
        variables = self._resolve_variables(ctx)

        if prompt.config:
            ctx.state["langfuse_prompt_config"] = prompt.config

        if self._prompt_type == "chat":
            self._apply_chat_prompt(ctx, prompt, variables)
        else:
            self._apply_text_prompt(ctx, prompt, variables)

        async for event in call_next(ctx):
            yield event

    def _apply_text_prompt(
        self,
        ctx: RequestContext,
        prompt: "TextPromptClient",
        variables: dict[str, str],
    ) -> None:
        compiled = prompt.compile(**variables) if variables else prompt.prompt
        existing = ctx.request.instructions or ""

        if self._position == "prepend":
            merged = f"{compiled}\n\n{existing}".strip()
        else:
            merged = f"{existing}\n\n{compiled}".strip()

        ctx.request = ctx.request.model_copy(update={"instructions": merged})

    def _apply_chat_prompt(
        self,
        ctx: RequestContext,
        prompt: "ChatPromptClient",
        variables: dict[str, str],
    ) -> None:
        compiled_messages = prompt.compile(**variables) if variables else prompt.prompt

        messages: list[InputMessage] = []
        for msg in compiled_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                existing = ctx.request.instructions or ""
                if self._position == "prepend":
                    merged = f"{content}\n\n{existing}".strip()
                else:
                    merged = f"{existing}\n\n{content}".strip()
                ctx.request = ctx.request.model_copy(update={"instructions": merged})
                continue

            messages.append(InputMessage(role=Role(role), content=content))

        if messages:
            existing_items = list(ctx.request.input)

            if self._position == "prepend":
                new_input = list(messages) + existing_items
            else:
                new_input = existing_items + list(messages)

            ctx.request = ctx.request.model_copy(update={"input": new_input})
