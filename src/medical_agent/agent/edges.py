from langchain_core.messages import AIMessage

from medical_agent.config import get_settings


def route_after_safety(state: dict) -> str:
    if state.get("is_emergency", False):
        return "emergency_response"
    return "agent"


def route_agent_or_tools(state: dict) -> str:
    """After agent node: route to tools if there are pending tool calls, else finish."""
    messages = state.get("messages", [])
    if not messages:
        return "save_memory"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "save_memory"


def route_after_save(state: dict) -> str:
    settings = get_settings()
    turn_count = state.get("turn_count", 0)
    if turn_count > 0 and turn_count % settings.summarize_every_n_turns == 0:
        return "summarize_conversation"
    return "__end__"
