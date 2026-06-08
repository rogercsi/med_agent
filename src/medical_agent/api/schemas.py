from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    patient_id: str = Field(..., description="患者唯一ID")
    session_id: str = Field(..., description="会话ID（对应LangGraph thread_id）")
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    rewritten_query: str
    sources: list[str]
    is_emergency: bool
    token_before_compress: int
    token_after_compress: int
    turn_count: int


class MemoryItem(BaseModel):
    memory: str
    id: str = ""


class MemoryResponse(BaseModel):
    patient_id: str
    memories: list[MemoryItem]


class EvalResult(BaseModel):
    mode: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    num_samples: int


class EvalCompareResponse(BaseModel):
    naive: EvalResult
    optimized: EvalResult
