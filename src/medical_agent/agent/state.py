from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PatientMemory(BaseModel):
    symptoms_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)

    def to_prompt_str(self) -> str:
        parts = []
        if self.allergies:
            parts.append(f"过敏史：{', '.join(self.allergies)}")
        if self.chronic_conditions:
            parts.append(f"慢性病史：{', '.join(self.chronic_conditions)}")
        if self.current_medications:
            parts.append(f"当前用药：{', '.join(self.current_medications)}")
        if self.symptoms_history:
            parts.append(f"历史症状：{', '.join(self.symptoms_history[-3:])}")
        if self.preferences:
            parts.append(f"患者偏好：{', '.join(self.preferences)}")
        return "\n".join(parts) if parts else "暂无记录"


ConsultationPhase = Literal[
    "greeting",
    "memory_inject",
    "intake",
    "safety_check",
    "retrieve",
    "compress",
    "respond",
    "save_memory",
    "summarize",
    "end",
]


class ConsultationState(dict):
    """LangGraph state as a TypedDict-compatible dict subclass."""

    messages: Annotated[list[BaseMessage], add_messages]
    patient_id: str
    session_id: str
    current_phase: ConsultationPhase
    turn_count: int
    is_emergency: bool
    emergency_keywords: list[str]
    patient_memory: PatientMemory
    rewritten_query: str
    retrieved_chunks: list[str]
    retrieved_sources: list[str]
    token_before_compress: int
    token_after_compress: int
    conversation_summary: str
    final_answer: str
