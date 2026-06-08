import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from medical_agent.config import get_settings


def print_comparison_table(data: dict | None = None) -> None:
    settings = get_settings()
    console = Console()

    if data is None:
        path = Path(settings.ragas_baseline_path)
        if not path.exists():
            console.print("[red]No baseline_results.json found. Run: uv run python scripts/ingest.py --eval[/red]")
            return
        with open(path) as f:
            data = json.load(f)

    naive = data.get("naive", {})
    optimized = data.get("optimized", {})

    table = Table(title="RAG Quality Comparison: Naive vs Optimized", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Naive RAG", justify="center")
    table.add_column("Optimized RAG", justify="center", style="green")
    table.add_column("Improvement", justify="center", style="cyan")

    metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall", "context_recall"),
    ]

    for label, key in metrics:
        n = naive.get(key, 0.0)
        o = optimized.get(key, 0.0)
        delta = ((o - n) / n * 100) if n > 0 else 0
        sign = "+" if delta >= 0 else ""
        table.add_row(label, f"{n:.3f}", f"{o:.3f}", f"{sign}{delta:.0f}%")

    console.print(table)
    console.print(
        f"\nSamples: naive={naive.get('num_samples', '?')}, "
        f"optimized={optimized.get('num_samples', '?')}"
    )
