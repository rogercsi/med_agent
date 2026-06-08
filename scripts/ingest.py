#!/usr/bin/env python
"""One-shot document ingestion: chunk → embed → upsert to Qdrant + build BM25 index."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.progress import Progress

console = Console()


def main(run_eval: bool = False) -> None:
    from medical_agent.config import get_settings
    from medical_agent.rag.indexer import run_ingest

    settings = get_settings()
    console.print("[bold blue]Medical Agent — Document Ingestion[/bold blue]")
    console.print(f"Docs path : {settings.raw_docs_path}")
    console.print(f"Qdrant    : {settings.qdrant_host}:{settings.qdrant_port}")
    console.print(f"Collection: {settings.qdrant_collection_rag}")

    with Progress() as progress:
        task = progress.add_task("Ingesting documents...", total=None)
        n_chunks = run_ingest(settings)
        progress.update(task, completed=1, total=1)

    console.print(f"[green]✓ Ingested {n_chunks} chunks into Qdrant and BM25 index[/green]")
    console.print(f"  BM25 index saved to: {settings.bm25_index_path}")

    if run_eval:
        console.print("\n[bold]Running Naive RAG evaluation (baseline)...[/bold]")
        asyncio.run(_run_eval_baseline(settings))


async def _run_eval_baseline(settings) -> None:
    from medical_agent.eval.ragas_runner import run_ragas_evaluation

    testset_path = Path(settings.ragas_testset_path)
    if not testset_path.exists():
        console.print(
            "[yellow]Testset not found. Run scripts/generate_testset.py first.[/yellow]"
        )
        return

    console.print("  Evaluating naive RAG...")
    naive_result = await run_ragas_evaluation(mode="naive")
    console.print("  Evaluating optimized RAG...")
    optimized_result = await run_ragas_evaluation(mode="optimized")

    output = {"naive": naive_result, "optimized": optimized_result}
    out_path = Path(settings.ragas_baseline_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Evaluation results saved to {out_path}[/green]")

    from medical_agent.eval.compare import print_comparison_table
    print_comparison_table(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest medical documents")
    parser.add_argument("--eval", action="store_true", help="Also run Ragas baseline eval")
    args = parser.parse_args()
    main(run_eval=args.eval)
