"""Tests for LangGraph agent state transitions and full graph invocation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from medical_agent.agent.state import PatientMemory


# ─── Unit: state serialization ──────────────────────────────────────────────

def test_patient_memory_to_prompt_str():
    pm = PatientMemory(
        allergies=["对青霉素过敏"],
        chronic_conditions=["2型糖尿病"],
        current_medications=["二甲双胍 500mg"],
    )
    s = pm.to_prompt_str()
    assert "青霉素" in s
    assert "糖尿病" in s
    assert "二甲双胍" in s


def test_patient_memory_empty():
    pm = PatientMemory()
    assert pm.to_prompt_str() == "暂无记录"


# ─── Unit: edge routing ──────────────────────────────────────────────────────

def test_route_after_safety_emergency():
    from medical_agent.agent.edges import route_after_safety
    assert route_after_safety({"is_emergency": True}) == "emergency_response"


def test_route_after_safety_normal():
    from medical_agent.agent.edges import route_after_safety
    assert route_after_safety({"is_emergency": False}) == "agent"


def test_route_agent_or_tools_with_tool_calls():
    from medical_agent.agent.edges import route_agent_or_tools
    msg = AIMessage(content="", tool_calls=[{"name": "search_medical_knowledge", "args": {}, "id": "1"}])
    assert route_agent_or_tools({"messages": [msg]}) == "tools"


def test_route_agent_or_tools_no_tool_calls():
    from medical_agent.agent.edges import route_agent_or_tools
    msg = AIMessage(content="建议就医")
    assert route_agent_or_tools({"messages": [msg]}) == "save_memory"


def test_route_after_save_no_summarize():
    from medical_agent.agent.edges import route_after_save
    with patch("medical_agent.agent.edges.get_settings") as mock_s:
        mock_s.return_value.summarize_every_n_turns = 6
        assert route_after_save({"turn_count": 3}) == "__end__"


def test_route_after_save_triggers_summarize():
    from medical_agent.agent.edges import route_after_save
    with patch("medical_agent.agent.edges.get_settings") as mock_s:
        mock_s.return_value.summarize_every_n_turns = 6
        assert route_after_save({"turn_count": 6}) == "summarize_conversation"


# ─── Unit: safety check node (keyword path, no LLM) ─────────────────────────

@pytest.mark.asyncio
async def test_node_safety_check_instant_emergency():
    """Instant-emergency keywords bypass LLM and always trigger."""
    from medical_agent.agent.nodes import node_safety_check

    state = {
        "messages": [HumanMessage(content="患者突然心脏骤停，完全失去意识")],
        "rewritten_query": "心脏骤停 意识丧失",
    }
    # Patch LLM so test never hits the network
    with patch("medical_agent.agent.nodes._make_chat_llm"):
        result = await node_safety_check(state)
    assert result["is_emergency"] is True


@pytest.mark.asyncio
async def test_node_safety_check_normal_with_llm_mock():
    """Non-emergency query: LLM structured output decides → mock returns False."""
    from medical_agent.agent.nodes import node_safety_check

    state = {
        "messages": [HumanMessage(content="我最近头痛，想了解一下高血压的治疗方法")],
        "rewritten_query": "高血压治疗",
    }
    mock_result = MagicMock()
    mock_result.is_emergency = False
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)

    with patch("medical_agent.agent.nodes._make_chat_llm", return_value=mock_llm):
        result = await node_safety_check(state)
    assert result["is_emergency"] is False


# ─── Integration: graph topology ─────────────────────────────────────────────

def test_graph_topology_nodes_and_edges():
    """Compile the graph and verify the expected node names are present."""
    from medical_agent.agent.graph import build_graph_definition
    from langgraph.checkpoint.memory import MemorySaver

    builder = build_graph_definition()
    graph = builder.compile(checkpointer=MemorySaver())

    node_names = set(graph.nodes)
    for expected in ("inject_memory", "intake", "safety_check",
                     "emergency_response", "agent", "tools",
                     "save_memory", "summarize_conversation"):
        assert expected in node_names, f"Missing node: {expected}"


# ─── Integration: full graph invocation ──────────────────────────────────────

@pytest.mark.asyncio
async def test_full_graph_normal_path(tmp_path):
    """Invoke the full graph on a non-emergency query with all I/O mocked."""
    from medical_agent.agent.graph import build_graph_definition
    from langgraph.checkpoint.memory import MemorySaver

    # --- mock external I/O ---
    # 1. Mem0 search/add
    mock_search = MagicMock(return_value=[])
    mock_add = MagicMock()

    # 2. AsyncOpenAI (used by node_intake and node_summarize_conversation)
    mock_async_openai = MagicMock()
    intake_resp = MagicMock()
    intake_resp.choices = [MagicMock(message=MagicMock(content="头痛发烧两天"))]
    mock_async_openai.return_value.chat.completions.create = AsyncMock(return_value=intake_resp)

    # 3. ChatOpenAI safety check (LLM structured output)
    mock_emergency_result = MagicMock()
    mock_emergency_result.is_emergency = False
    mock_chat_llm = MagicMock()
    mock_chat_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=mock_emergency_result
    )

    # 4. ChatOpenAI agent node (streaming=True, bind_tools) → final answer, no tool calls
    final_ai_msg = AIMessage(content="根据症状建议就医，保持休息。")
    mock_chat_llm_streaming = MagicMock()
    mock_chat_llm_streaming.bind_tools.return_value.ainvoke = AsyncMock(return_value=final_ai_msg)

    def _mock_make_chat_llm(streaming: bool = False) -> MagicMock:
        return mock_chat_llm_streaming if streaming else mock_chat_llm

    builder = build_graph_definition()
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-session-001"}}

    initial_state = {
        "messages": [HumanMessage(content="我头痛发烧两天了")],
        "patient_id": "test-p001",
        "session_id": "test-session-001",
        "current_phase": "greeting",
        "turn_count": 0,
        "is_emergency": False,
        "emergency_keywords": [],
        "patient_memory": {},
        "rewritten_query": "",
        "retrieved_chunks": [],
        "retrieved_sources": [],
        "token_before_compress": 0,
        "token_after_compress": 0,
        "conversation_summary": "",
        "final_answer": "",
    }

    with patch("medical_agent.agent.nodes.search_memory", mock_search), \
         patch("medical_agent.agent.nodes.add_memory", mock_add), \
         patch("medical_agent.agent.nodes.AsyncOpenAI", mock_async_openai), \
         patch("medical_agent.agent.nodes._make_chat_llm", side_effect=_mock_make_chat_llm):

        final_state = await graph.ainvoke(initial_state, config=config)

    # Graph completed and produced a final answer
    assert "messages" in final_state
    assert any(
        isinstance(m, AIMessage) and "就医" in (m.content or "")
        for m in final_state["messages"]
    )
    assert final_state.get("is_emergency") is False
    mock_search.assert_called_once()  # memory was queried


@pytest.mark.asyncio
async def test_full_graph_emergency_path():
    """Emergency path: safety_check fires → emergency_response node, no tool loop."""
    from medical_agent.agent.graph import build_graph_definition
    from langgraph.checkpoint.memory import MemorySaver

    mock_search = MagicMock(return_value=[])
    mock_add = MagicMock()

    intake_resp = MagicMock()
    intake_resp.choices = [MagicMock(message=MagicMock(content="压迫性胸痛放射至左臂"))]
    mock_async_openai = MagicMock()
    mock_async_openai.return_value.chat.completions.create = AsyncMock(return_value=intake_resp)

    builder = build_graph_definition()
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-emergency-001"}}

    initial_state = {
        "messages": [HumanMessage(content="我突然感到剧烈压迫性胸痛，向左臂放射，大汗，心脏骤停感")],
        "patient_id": "test-p002",
        "session_id": "test-emergency-001",
        "current_phase": "greeting",
        "turn_count": 0,
        "is_emergency": False,
        "emergency_keywords": [],
        "patient_memory": {},
        "rewritten_query": "",
        "retrieved_chunks": [],
        "retrieved_sources": [],
        "token_before_compress": 0,
        "token_after_compress": 0,
        "conversation_summary": "",
        "final_answer": "",
    }

    with patch("medical_agent.agent.nodes.search_memory", mock_search), \
         patch("medical_agent.agent.nodes.add_memory", mock_add), \
         patch("medical_agent.agent.nodes.AsyncOpenAI", mock_async_openai):

        final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state.get("is_emergency") is True
    assert "120" in final_state.get("final_answer", "")
