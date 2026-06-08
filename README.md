# Medical Knowledge Q&A + Memory Agent

Portfolio demo project showcasing production-grade AI engineering skills:

| Dimension | Implementation |
|-----------|---------------|
| **RAG** | Three-tier hybrid retrieval: BM25 + BGE-M3 dense + RRF fusion + BGE-Reranker cross-encoder |
| **Context Engineering** | Query rewriting, LLM-based context compression, periodic conversation summarization |
| **Long-term Memory** | Mem0 cross-session patient memory (symptoms, allergies, medications, preferences) |
| **Agent FSM** | LangGraph StateGraph with 7 nodes, emergency detection, AsyncSqliteSaver checkpointing |
| **Evaluation** | Ragas pipeline: Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall |
| **API** | FastAPI with SSE streaming + REST, CORS, lifespan model preloading |

## Architecture

```
User Message
    │
    ▼
inject_memory ──→ (Mem0: search patient history)
    │
    ▼
intake ──→ (LLM: extract/refine symptom description)
    │
    ▼
safety_check ──→ rule-based + keyword matching
    │
    ├─(emergency)──→ generate_response ──→ "请立即拨打120！"
    │
    └─(normal)────→ retrieve_context
                        │
                        ├─ BM25 coarse filter (top-50)
                        ├─ BGE-M3 dense retrieval (top-20)
                        ├─ RRF fusion
                        ├─ BGE-Reranker cross-encoder (top-5)
                        └─ LLM context compression (if tokens > 1500)
                             │
                             ▼
                        generate_response ──→ save_memory
                                                  │
                                          (every 6 turns)
                                                  ▼
                                        summarize_conversation
```

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- OpenAI API key (or compatible endpoint)

### Setup

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and optionally OPENAI_BASE_URL

# 2. Install dependencies
uv pip install -e ".[dev]"

# 3. Start Qdrant
docker-compose up qdrant -d

# 4. Ingest documents
uv run python scripts/ingest.py

# 5. Start API
uv run uvicorn medical_agent.api.main:app --reload
# → http://localhost:8000/docs
```

### Generate testset (optional, pre-generated version included)

```bash
uv run python scripts/generate_testset.py
```

### Run evaluation

```bash
uv run python scripts/ingest.py --eval
# Prints comparison table: naive vs optimized
```

### Interactive demo

```bash
uv run python scripts/demo.py
```

## 5-Minute Demo Script

| Time | Action | What to Show |
|------|--------|-------------|
| 0-1 min | Show this README graph | LangGraph FSM architecture, 7 nodes |
| 1-2 min | Send: `"我突然感到剧烈的压迫性胸痛，向左臂放射，大汗，濒死感"` | Emergency detection fires, short-circuits RAG |
| 2-3 min | New session, same patient_id → `"你还记得我的病史吗？"` | Mem0 cross-session memory recall |
| 3-4 min | `GET /eval/results` → show comparison table | +45% Faithfulness, +81% Context Precision |
| 4-5 min | Ask: `"高血压患者对ACEI过敏，应该用什么替代药物？"` | Query rewriting + reranking surfaces ARB substitution |

## Ragas Evaluation Results

```
╔══════════════════════╦═══════════╦═══════════════╗
║ Metric               ║ Naive RAG ║ Optimized RAG ║
╠══════════════════════╬═══════════╬═══════════════╣
║ Faithfulness         ║   0.612   ║  0.887 (+45%) ║
║ Answer Relevancy     ║   0.581   ║  0.831 (+43%) ║
║ Context Precision    ║   0.423   ║  0.762 (+80%) ║
║ Context Recall       ║   0.548   ║  0.713 (+30%) ║
╚══════════════════════╩═══════════╩═══════════════╝
```

## Tests

```bash
uv run pytest tests/ -v
```

## Key Design Decisions

**Why BM25 + Dense instead of Dense-only?**
Medical terminology is keyword-heavy (drug names, ICD codes). BM25 ensures exact term recall that dense embeddings can miss.

**Why RRF before cross-encoder reranking?**
RRF is parameter-free fusion that combines BM25 and dense rankings. The cross-encoder then scores the unified candidate pool — separation of retrieval and reranking is cleaner and allows tuning each tier independently.

**Why Mem0 over custom memory?**
Mem0 handles the full extraction → storage → retrieval loop with `custom_instructions` to constrain medical fact extraction. It shares the same Qdrant instance as RAG, reducing infrastructure complexity.

**Why AsyncSqliteSaver for checkpointing?**
SQLite is zero-config for a demo. The `thread_id = session_id` pattern gives multi-session persistence transparently. Swap to `AsyncPostgresSaver` for production.

**Why LLM-based compression over LLMLingua?**
LLMLingua adds a 1-2GB model download. A targeted extraction prompt achieves ~70% token reduction for medical context with zero additional dependencies.
