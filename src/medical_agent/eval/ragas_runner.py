import json
from pathlib import Path
from typing import Literal

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from medical_agent.config import get_settings
from medical_agent.rag.pipeline import get_rag_pipeline


def _load_testset(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("samples", [])


async def _answer_question(query: str, chunks: list[str]) -> str:
    from openai import AsyncOpenAI
    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    context = "\n\n".join(chunks) or "无相关知识"
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": f"根据以下医疗知识回答问题，保持简洁：\n\n{context}",
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


async def run_ragas_evaluation(
    mode: Literal["naive", "optimized"] = "optimized",
) -> dict:
    settings = get_settings()
    testset = _load_testset(settings.ragas_testset_path)

    if not testset:
        raise ValueError(f"Empty testset at {settings.ragas_testset_path}")

    pipeline = get_rag_pipeline()
    rows = []

    for item in testset:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        ctx = pipeline.query(question, mode=mode)
        answer = await _answer_question(question, ctx.chunks)

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": ctx.chunks if ctx.chunks else [""],
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_list(rows)

    scores = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
    )

    scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

    return {
        "mode": mode,
        "faithfulness": round(scores_dict.get("faithfulness", 0.0), 4),
        "answer_relevancy": round(scores_dict.get("answer_relevancy", 0.0), 4),
        "context_precision": round(scores_dict.get("context_precision", 0.0), 4),
        "context_recall": round(scores_dict.get("context_recall", 0.0), 4),
        "num_samples": len(rows),
    }
