import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.logging import logger
from src.config.settings import settings


def compute_config_hash(overrides: dict[str, Any] | None = None) -> str:
    """Computes SHA-256 fingerprint of current pipeline configuration."""
    cfg = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
        "groq_model": settings.groq_model,
        "temperature": settings.temperature,
    }
    if overrides:
        cfg.update(overrides)

    encoded = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


class MetricsStore:
    """Persists historical RAGAS experiment runs to eval/results as permanent artifacts."""

    def __init__(self, results_dir: str = "eval/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_run(
        self,
        run_id: str,
        config_hash: str,
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "run_id": run_id,
            "config_hash": config_hash,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        file_path = self.results_dir / f"{run_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Saved evaluation experiment artifact: {file_path}")
        return record

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        file_path = self.results_dir / f"{run_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for file in sorted(self.results_dir.glob("*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception as e:
                logger.error(f"Error reading result file {file}: {e}")
        return runs


metrics_store = MetricsStore()


def get_eval_result(run_id: str) -> dict[str, Any] | None:
    return metrics_store.get_run(run_id)


def run_eval_pipeline(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Runs evaluation or stores placeholder result if dataset is pending."""
    from src.eval.ragas_runner import run_ragas_evaluation

    run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    config_hash = compute_config_hash(config_overrides)

    metrics = run_ragas_evaluation(dataset_name="default_eval.json")
    record = metrics_store.save_run(
        run_id=run_id,
        config_hash=config_hash,
        metrics=metrics,
    )
    return {
        "run_id": run_id,
        "status": "completed",
        "config_hash": config_hash,
    }
