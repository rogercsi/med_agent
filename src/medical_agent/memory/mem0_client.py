from functools import lru_cache

from mem0 import Memory

from medical_agent.config import get_settings

_MEM0_CUSTOM_INSTRUCTIONS = (
    "仅提取医疗事实：症状、诊断、过敏史（包括药物过敏）、当前用药、慢性病史、患者偏好。"
    "格式：'<主语> <谓语> <宾语>'，例如：'患者 对青霉素 过敏'。"
    "忽略问候、一般咨询问题和非医疗内容。"
)


def _build_config() -> dict:
    settings = get_settings()
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings.llm_model,
                "api_key": settings.openai_api_key,
                "openai_base_url": settings.openai_base_url,
                "max_tokens": 1000,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": settings.embed_model,
                "embedding_dims": 1024,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": settings.qdrant_collection_mem0,
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "embedding_model_dims": 1024,
            },
        },
        "custom_prompt": _MEM0_CUSTOM_INSTRUCTIONS,
    }


@lru_cache(maxsize=1)
def get_memory() -> Memory:
    return Memory.from_config(_build_config())


def add_memory(messages: list[dict], patient_id: str) -> None:
    """Persist facts extracted from a conversation turn."""
    memory = get_memory()
    memory.add(messages, user_id=patient_id)


def search_memory(query: str, patient_id: str, limit: int = 10) -> list[dict]:
    """Retrieve relevant memories for a patient.

    mem0ai uses `filters={"user_id": ...}` in search(), not `user_id=` directly.
    """
    memory = get_memory()
    results = memory.search(query, filters={"user_id": patient_id}, limit=limit)
    if isinstance(results, dict):
        return results.get("results", [])
    return results or []


def get_all_memories(patient_id: str) -> list[dict]:
    """Return all stored memories for a patient (for demo display)."""
    memory = get_memory()
    results = memory.get_all(filters={"user_id": patient_id})
    if isinstance(results, dict):
        return results.get("results", [])
    return results or []
