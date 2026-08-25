import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    contexts: list[str] | None = None


class EvalDatasetLoader:
    """Loads evaluation datasets from eval/datasets directory."""

    def __init__(self, datasets_dir: str = "eval/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, filename: str = "default_eval.json") -> list[EvalSample]:
        file_path = self.datasets_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset '{filename}' not found in {self.datasets_dir}. "
                f"Please place your 50-60 ground truth evaluation questions in {self.datasets_dir}."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples: list[EvalSample] = []
        for item in data:
            samples.append(
                EvalSample(
                    question=item.get("question", ""),
                    ground_truth=item.get("ground_truth", ""),
                    contexts=item.get("contexts", []),
                )
            )
        return samples


dataset_loader = EvalDatasetLoader()
