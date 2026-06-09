import json

from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from medical_agent.agent.graph import get_compiled_graph, get_thread_config
from medical_agent.api.schemas import ChatRequest, ChatResponse, MemoryItem, MemoryResponse
from medical_agent.memory.mem0_client import get_all_memories

router = APIRouter(prefix="/chat", tags=["chat"])

_GRAPH_NODES = {
    "inject_memory", "intake", "safety_check",
    "emergency_response", "agent", "tools",
    "save_memory", "summarize_conversation",
}

_INITIAL_STATE_DEFAULTS = {
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


def _build_initial_state(req: ChatRequest) -> dict:
    return {
        **_INITIAL_STATE_DEFAULTS,
        "messages": [HumanMessage(content=req.message)],
        "patient_id": req.patient_id,
        "session_id": req.session_id,
    }


@router.post("/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest) -> ChatResponse:
    graph = await get_compiled_graph()
    config = get_thread_config(req.session_id)
    final_state = await graph.ainvoke(_build_initial_state(req), config=config)

    # Extract final answer from messages if not set by emergency node
    answer = final_state.get("final_answer", "")
    if not answer:
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                answer = msg.content or ""
                break

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        rewritten_query=final_state.get("rewritten_query", ""),
        sources=final_state.get("retrieved_sources", []),
        is_emergency=final_state.get("is_emergency", False),
        token_before_compress=final_state.get("token_before_compress", 0),
        token_after_compress=final_state.get("token_after_compress", 0),
        turn_count=final_state.get("turn_count", 0),
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming: real token-by-token output + tool call events + node progress."""

    graph = await get_compiled_graph()
    config = get_thread_config(req.session_id)

    async def event_generator():
        try:
            async for event in graph.astream_events(
                _build_initial_state(req), config=config, version="v2"
            ):
                event_type = event.get("event", "")
                name = event.get("name", "")

                # Node lifecycle events
                if event_type == "on_chain_start" and name in _GRAPH_NODES:
                    yield {
                        "event": "node_start",
                        "data": json.dumps({"node": name}),
                    }

                # Real token streaming from the LLM inside node_agent_with_tools
                elif event_type == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        yield {
                            "event": "token",
                            "data": json.dumps({"text": chunk.content}),
                        }

                # Tool call dispatched by the agent
                elif event_type == "on_tool_start":
                    yield {
                        "event": "tool_call",
                        "data": json.dumps({
                            "tool": name,
                            "input": event["data"].get("input", {}),
                        }),
                    }

                # Tool result returned to the agent
                elif event_type == "on_tool_end":
                    output = event["data"].get("output", "")
                    yield {
                        "event": "tool_result",
                        "data": json.dumps({
                            "tool": name,
                            "output": str(output)[:500],
                        }),
                    }

                # RAG metadata after retrieve_context equivalent (from agent tool call)
                elif event_type == "on_chain_end" and name == "safety_check":
                    output = event.get("data", {}).get("output", {})
                    if output.get("is_emergency"):
                        yield {
                            "event": "emergency",
                            "data": json.dumps({"keywords": output.get("emergency_keywords", [])}),
                        }

            yield {"event": "done", "data": "{}"}

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())


@router.get("/memory/{patient_id}", response_model=MemoryResponse)
async def get_patient_memory(patient_id: str) -> MemoryResponse:
    memories = get_all_memories(patient_id)
    items = [
        MemoryItem(memory=m.get("memory", ""), id=str(m.get("id", "")))
        for m in memories
    ]
    return MemoryResponse(patient_id=patient_id, memories=items)
