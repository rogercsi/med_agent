import asyncio
from functools import partial

from langchain_core.tools import tool

from medical_agent.rag.pipeline import get_rag_pipeline


@tool
async def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base for information about symptoms, conditions, or treatments.
    Use this for any medical question that needs evidence-based information."""
    pipeline = get_rag_pipeline()
    result = await asyncio.to_thread(partial(pipeline.query, query, "optimized"))
    if not result.chunks:
        return "未找到相关医疗知识，建议咨询专业医生。"
    return "\n\n---\n\n".join(result.chunks[:3])


@tool
async def check_drug_interaction(drug_names: str) -> str:
    """Check potential drug interactions or contraindications.
    Input: comma-separated drug names (e.g. '阿司匹林, 华法林')."""
    pipeline = get_rag_pipeline()
    result = await asyncio.to_thread(
        partial(pipeline.query, f"药物相互作用 禁忌 {drug_names}", "optimized")
    )
    if not result.chunks:
        return f"未找到 {drug_names} 相关的药物相互作用信息，建议查阅药品说明书或咨询药剂师。"
    return "\n\n---\n\n".join(result.chunks[:3])


MEDICAL_TOOLS = [search_medical_knowledge, check_drug_interaction]
