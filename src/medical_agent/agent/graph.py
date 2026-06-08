from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from medical_agent.agent.edges import route_after_save, route_after_safety
from medical_agent.agent.nodes import (
    node_generate_response,
    node_inject_memory,
    node_intake,
    node_retrieve_context,
    node_safety_check,
    node_save_memory,
    node_summarize_conversation,
)
from medical_agent.agent.state import ConsultationState
from medical_agent.config import get_settings


def build_graph_definition() -> StateGraph:
    builder = StateGraph(ConsultationState)

    builder.add_node("inject_memory", node_inject_memory)
    builder.add_node("intake", node_intake)
    builder.add_node("safety_check", node_safety_check)
    builder.add_node("retrieve_context", node_retrieve_context)
    builder.add_node("generate_response", node_generate_response)
    builder.add_node("save_memory", node_save_memory)
    builder.add_node("summarize_conversation", node_summarize_conversation)

    builder.add_edge(START, "inject_memory")
    builder.add_edge("inject_memory", "intake")
    builder.add_edge("intake", "safety_check")

    builder.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {
            "retrieve_context": "retrieve_context",
            "generate_response": "generate_response",
        },
    )

    builder.add_edge("retrieve_context", "generate_response")
    builder.add_edge("generate_response", "save_memory")

    builder.add_conditional_edges(
        "save_memory",
        route_after_save,
        {
            "summarize_conversation": "summarize_conversation",
            "__end__": END,
        },
    )

    builder.add_edge("summarize_conversation", END)

    return builder


_GRAPH_INSTANCE = None


async def get_compiled_graph():
    """Return a compiled graph with async SQLite checkpointer."""
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is not None:
        return _GRAPH_INSTANCE

    settings = get_settings()
    builder = build_graph_definition()

    checkpointer = AsyncSqliteSaver.from_conn_string(settings.checkpoints_db_path)
    _GRAPH_INSTANCE = builder.compile(checkpointer=checkpointer)
    return _GRAPH_INSTANCE


def get_thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}
