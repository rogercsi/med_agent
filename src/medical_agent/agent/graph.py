from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from medical_agent.agent.edges import route_after_save, route_after_safety, route_agent_or_tools
from medical_agent.agent.nodes import (
    node_agent_with_tools,
    node_emergency_response,
    node_inject_memory,
    node_intake,
    node_safety_check,
    node_save_memory,
    node_summarize_conversation,
)
from medical_agent.agent.tools import MEDICAL_TOOLS
from medical_agent.agent.state import ConsultationState
from medical_agent.config import get_settings


def build_graph_definition() -> StateGraph:
    """
    Graph topology:
    START → inject_memory → intake → safety_check
        ├─(emergency) → emergency_response → save_memory
        └─(normal) → agent ⇄ tools (ReAct loop)
                         └─(no tool_calls) → save_memory
                                                 ├─(every N turns) → summarize_conversation → END
                                                 └─(otherwise) → END
    """
    builder = StateGraph(ConsultationState)

    builder.add_node("inject_memory", node_inject_memory)
    builder.add_node("intake", node_intake)
    builder.add_node("safety_check", node_safety_check)
    builder.add_node("emergency_response", node_emergency_response)
    builder.add_node("agent", node_agent_with_tools)
    builder.add_node("tools", ToolNode(MEDICAL_TOOLS))
    builder.add_node("save_memory", node_save_memory)
    builder.add_node("summarize_conversation", node_summarize_conversation)

    builder.add_edge(START, "inject_memory")
    builder.add_edge("inject_memory", "intake")
    builder.add_edge("intake", "safety_check")

    builder.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {
            "emergency_response": "emergency_response",
            "agent": "agent",
        },
    )

    builder.add_edge("emergency_response", "save_memory")

    # ReAct tool-call loop
    builder.add_conditional_edges(
        "agent",
        route_agent_or_tools,
        {
            "tools": "tools",
            "save_memory": "save_memory",
        },
    )
    builder.add_edge("tools", "agent")

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
