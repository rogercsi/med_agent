import json

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse

from medical_agent.agent.graph import get_compiled_graph, get_thread_config
from medical_agent.api.schemas import ChatRequest, ChatResponse, MemoryResponse, MemoryItem
from medical_agent.memory.mem0_client import get_all_memories

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest) -> ChatResponse:
    graph = await get_compiled_graph()
    config = get_thread_config(req.session_id)

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "patient_id": req.patient_id,
        "session_id": req.session_id,
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

    final_state = await graph.ainvoke(initial_state, config=config)

    return ChatResponse(
        session_id=req.session_id,
        answer=final_state.get("final_answer", ""),
        rewritten_query=final_state.get("rewritten_query", ""),
        sources=final_state.get("retrieved_sources", []),
        is_emergency=final_state.get("is_emergency", False),
        token_before_compress=final_state.get("token_before_compress", 0),
        token_after_compress=final_state.get("token_after_compress", 0),
        turn_count=final_state.get("turn_count", 0),
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint — emits node progress events + final answer tokens."""

    graph = await get_compiled_graph()
    config = get_thread_config(req.session_id)

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "patient_id": req.patient_id,
        "session_id": req.session_id,
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

    async def event_generator():
        try:
            async for event in graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                event_type = event.get("event", "")
                name = event.get("name", "")

                if event_type == "on_chain_start" and name in {
                    "inject_memory", "intake", "safety_check",
                    "retrieve_context", "generate_response",
                    "save_memory", "summarize_conversation",
                }:
                    yield {
                        "event": "node_start",
                        "data": json.dumps({"node": name}),
                    }

                elif event_type == "on_chain_end" and name == "retrieve_context":
                    output = event.get("data", {}).get("output", {})
                    yield {
                        "event": "rag_result",
                        "data": json.dumps({
                            "rewritten_query": output.get("rewritten_query", ""),
                            "sources": output.get("retrieved_sources", []),
                            "token_before": output.get("token_before_compress", 0),
                            "token_after": output.get("token_after_compress", 0),
                        }),
                    }

                elif event_type == "on_chain_end" and name == "safety_check":
                    output = event.get("data", {}).get("output", {})
                    if output.get("is_emergency"):
                        yield {
                            "event": "emergency",
                            "data": json.dumps({
                                "keywords": output.get("emergency_keywords", [])
                            }),
                        }

                elif event_type == "on_chain_end" and name == "generate_response":
                    output = event.get("data", {}).get("output", {})
                    yield {
                        "event": "answer",
                        "data": json.dumps({"text": output.get("final_answer", "")}),
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
