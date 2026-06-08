#!/usr/bin/env python
"""Generate Ragas synthetic testset from medical documents (run once, commit result)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


def main() -> None:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from llama_index.core import SimpleDirectoryReader
    from ragas.testset import TestsetGenerator

    from medical_agent.config import get_settings

    settings = get_settings()

    console.print("[bold blue]Generating Ragas synthetic testset...[/bold blue]")
    console.print(f"Source docs: {settings.raw_docs_path}")
    console.print(f"LLM model  : {settings.llm_model}")

    docs = SimpleDirectoryReader(settings.raw_docs_path).load_data()
    console.print(f"Loaded {len(docs)} documents")

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    generator = TestsetGenerator.from_llama_index(
        llm=llm,
        embedding_model=embeddings,
    )

    testset = generator.generate_with_llama_index_docs(
        documents=docs,
        testset_size=30,
    )

    out_path = Path(settings.ragas_testset_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for row in testset.to_pandas().to_dict(orient="records"):
        samples.append({
            "question": row.get("question", row.get("user_input", "")),
            "ground_truth": row.get("ground_truth", row.get("reference", "")),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Generated {len(samples)} QA pairs → {out_path}[/green]")


if __name__ == "__main__":
    main()
