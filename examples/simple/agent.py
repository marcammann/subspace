from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from subspace import RequestContext

in_memory_saver = InMemorySaver()


@tool
def get_weather(query: str) -> str:
    """Get Weather"""
    resp = interrupt("Whoops")
    return resp


def make_agent(ctx: RequestContext, interrupt_tools: list):
    return create_agent(
        "anthropic:claude-sonnet-4-6",
        system_prompt=ctx.request.instructions or "You are a helpful assistant. Be concise.",
        tools=[get_weather] + interrupt_tools,
        checkpointer=in_memory_saver,
    )
