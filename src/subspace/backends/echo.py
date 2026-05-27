from collections.abc import AsyncIterator

from subspace.backends.response_builder import ResponseBuilder
from subspace.middleware.context import RequestContext
from subspace.models.common import Usage
from subspace.models.content import InputTextContent
from subspace.models.events import StreamEvent
from subspace.models.items import InputMessage


class EchoBackend:
    async def handle(self, ctx: RequestContext) -> AsyncIterator[StreamEvent]:
        text = _extract_text(ctx)
        builder = ResponseBuilder(ctx.response)

        for event in builder.start():
            yield event

        for word in text.split():
            for event in builder.text_delta(word + " "):
                yield event

        word_count = len(text.split())
        builder.set_usage(
            Usage(
                input_tokens=word_count,
                output_tokens=word_count,
                total_tokens=word_count * 2,
            )
        )

        for event in builder.finish():
            yield event


def _extract_text(ctx: RequestContext) -> str:
    for item in ctx.request.input:
        if isinstance(item, InputMessage):
            if isinstance(item.content, str):
                return item.content
            for part in item.content:
                if isinstance(part, InputTextContent):
                    return part.text
    return "echo"
