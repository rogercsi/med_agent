import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException

from medical_agent.api.schemas import EvalCompareResponse, EvalResult
from medical_agent.config import get_settings

router = APIRouter(prefix="/eval", tags=["eval"])

_running_eval: dict[str, bool] = {}


@router.get("/results", response_model=EvalCompareResponse)
async def get_eval_results() -> EvalCompareResponse:
    """Return pre-committed baseline + current optimized results."""
    settings = get_settings()
    path = Path(settings.ragas_baseline_path)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Baseline results not found. Run: uv run python scripts/ingest.py --eval",
        )

    with open(path) as f:
        data = json.load(f)

    return EvalCompareResponse(
        naive=EvalResult(**data["naive"]),
        optimized=EvalResult(**data["optimized"]),
    )


@router.post("/run")
async def run_eval(
    mode: Literal["naive", "optimized"] = "optimized",
    background_tasks: BackgroundTasks = None,
) -> dict:
    """Trigger a live Ragas evaluation (runs in background)."""
    if _running_eval.get(mode):
        return {"status": "already_running", "mode": mode}

    _running_eval[mode] = True

    async def _run():
        try:
            from medical_agent.eval.ragas_runner import run_ragas_evaluation
            result = await run_ragas_evaluation(mode=mode)
            settings = get_settings()
            path = Path(settings.ragas_baseline_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if path.exists():
                with open(path) as f:
                    existing = json.load(f)
            existing[mode] = result
            with open(path, "w") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        finally:
            _running_eval[mode] = False

    if background_tasks:
        background_tasks.add_task(_run)
    else:
        await _run()

    return {"status": "started", "mode": mode}
