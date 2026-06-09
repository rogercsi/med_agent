from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medical_agent.api.routes.chat import router as chat_router
from medical_agent.api.routes.eval import router as eval_router
from medical_agent.observability import RequestLoggingMiddleware, configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # Warm up: pre-load models and compile graph so first request is fast
    from medical_agent.agent.graph import get_compiled_graph
    from medical_agent.rag.embedder import get_embedder
    from medical_agent.rag.reranker import get_reranker

    get_embedder()
    get_reranker()
    await get_compiled_graph()
    yield


app = FastAPI(
    title="Medical Knowledge Q&A + Memory Agent",
    description=(
        "Showcases: three-tier hybrid RAG (BM25 + BGE-M3 + Reranker), "
        "Mem0 cross-session patient memory, LangGraph FSM with tool-calling agent, "
        "real SSE token streaming, and Ragas evaluation pipeline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(eval_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
