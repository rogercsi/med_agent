from medical_agent.config import get_settings


def route_after_safety(state: dict) -> str:
    if state.get("is_emergency", False):
        return "generate_response"  # emergency path skips RAG
    return "retrieve_context"


def route_after_response(state: dict) -> str:
    return "save_memory"


def route_after_save(state: dict) -> str:
    settings = get_settings()
    turn_count = state.get("turn_count", 0)
    if turn_count > 0 and turn_count % settings.summarize_every_n_turns == 0:
        return "summarize_conversation"
    return "__end__"
