import csv
from pathlib import Path
from typing import Dict, List


def parse_results_csv(path: Path) -> Dict[str, List[Dict[str, float]]]:
    if not path.exists():
        return {}
    metrics: Dict[str, List[Dict[str, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row.get("epoch", 0))
            for key, value in row.items():
                if key == "epoch":
                    continue
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                metrics.setdefault(key, []).append({"x": epoch, "y": val})
    return metrics
