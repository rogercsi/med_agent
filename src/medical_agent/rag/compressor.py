import tiktoken
from openai import OpenAI

from medical_agent.config import get_settings


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def compress_context(chunks: list[str], query: str) -> list[str]:
    """Selectively extract only query-relevant sentences via LLM."""
    settings = get_settings()
    total_tokens = sum(_count_tokens(c) for c in chunks)

    if total_tokens <= settings.compress_token_threshold:
        return chunks

    combined = "\n\n---\n\n".join(chunks)
    prompt = (
        f"以下是医疗知识上下文，请仅提取与问题「{query}」直接相关的句子。"
        "保留原文表述，不做解释，不做改写。用换行分隔提取的句子。\n\n"
        f"上下文：\n{combined}"
    )

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800,
    )
    compressed = resp.choices[0].message.content or ""
    return [compressed] if compressed.strip() else chunks
