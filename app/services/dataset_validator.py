import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from app.models import (
    DatasetConfig,
    DatasetStats,
    DatasetValidationResult,
    ValidationMessage,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _resolve(path: Optional[str], base: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute() and base:
        p = Path(base) / p
    return p


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _sample_images(directory: Path, limit: int = 5) -> List[Path]:
    images = [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return images[:limit]


def _check_image_readable(path: Path) -> bool:
    try:
        import cv2

        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        # If cv2 is not available, skip the readability test.
        return True


def _validate_label_file(path: Path, task_type: str) -> bool:
    try:
        content = path.read_text().strip().splitlines()
    except Exception:
        return False
    if not content:
        return False
    if task_type == "classify":
        return True
    if task_type == "detect":
        for line in content[:5]:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            try:
                [float(x) for x in parts]
            except ValueError:
                return False
    else:  # segment
        # YOLO segment labels have class + at least 6 coords (3 points)
        for line in content[:5]:
            parts = line.strip().split()
            if len(parts) < 7:
                return False
            try:
                [float(x) for x in parts]
            except ValueError:
                return False
    return True


def _infer_class_names_from_dirs(directory: Path) -> List[str]:
    """Infer class names from immediate subdirectories (classification)."""
    if not directory.exists():
        return []
    return sorted([p.name for p in directory.iterdir() if p.is_dir()])


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_dataset(config: DatasetConfig) -> DatasetValidationResult:
    messages: List[ValidationMessage] = []
    stats = DatasetStats()

    root = Path(config.root)
    if not root.exists():
        return DatasetValidationResult(
            ok=False,
            messages=[ValidationMessage(level="error", text=f"Root not found: {root}")],
        )

    train_dir: Optional[Path] = None
    val_dir: Optional[Path] = None
    test_dir: Optional[Path] = None
    names: List[str] = []
    nc: Optional[int] = None

    # Classification: prefer folder structure, no label files.
    if config.task_type == "classify" and not config.data_yaml:
        train_dir = _resolve(config.train_dir, root) or root / "train"
        val_dir = _resolve(config.val_dir, root) or root / "val"
        test_dir = _resolve(config.test_dir, root) or root / "test"
        for key, d in [("train", train_dir), ("val", val_dir)]:
            if not d or not d.exists():
                messages.append(
                    ValidationMessage(level="error", text=f"{key} directory missing: {d}")
                )
        if train_dir and train_dir.exists():
            names = _infer_class_names_from_dirs(train_dir)
            if not names:
                messages.append(
                    ValidationMessage(
                        level="warning",
                        text="No class subdirectories found under train; classification labels may be missing.",
                    )
                )
            nc = len(names)
        messages.append(
            ValidationMessage(level="info", text=f"Using folder-based classification under {root}")
        )
    elif config.data_yaml:
        data_path = _resolve(config.data_yaml, root)
        if not data_path or not data_path.exists():
            return DatasetValidationResult(
                ok=False,
                messages=[
                    ValidationMessage(
                        level="error", text=f"data.yaml not found: {data_path}"
                    )
                ],
            )
        data_cfg = _load_yaml(data_path)
        for key in ["train", "val"]:
            if key not in data_cfg:
                messages.append(
                    ValidationMessage(level="error", text=f"Missing key: {key}")
                )
        train_dir = _resolve(data_cfg.get("train"), data_path.parent)
        val_dir = _resolve(data_cfg.get("val"), data_path.parent)
        test_dir = _resolve(data_cfg.get("test"), data_path.parent)
        nc = data_cfg.get("nc")
        names = data_cfg.get("names", []) or []
        if nc is not None and names and len(names) != nc:
            messages.append(
                ValidationMessage(
                    level="warning",
                    text=f"nc ({nc}) does not match names length ({len(names)})",
                )
            )
        if nc and not names:
            names = [str(i) for i in range(nc)]
        messages.append(
            ValidationMessage(
                level="info",
                text=f"Loaded data.yaml from {data_path}",
            )
        )
    else:
        train_dir = _resolve(config.train_dir, root)
        val_dir = _resolve(config.val_dir, root)
        test_dir = _resolve(config.test_dir, root)
        if not train_dir or not train_dir.exists():
            messages.append(
                ValidationMessage(
                    level="error", text="Train directory missing or not provided"
                )
            )
        if not val_dir or not val_dir.exists():
            messages.append(
                ValidationMessage(level="error", text="Val directory missing")
            )

    # Count images
    if train_dir and train_dir.exists():
        stats.train_images = _count_images(train_dir)
        if stats.train_images == 0:
            messages.append(
                ValidationMessage(level="error", text="No train images found")
            )
    if val_dir and val_dir.exists():
        stats.val_images = _count_images(val_dir)
        if stats.val_images == 0:
            messages.append(
                ValidationMessage(level="warning", text="No val images found")
            )
    if test_dir and test_dir.exists():
        stats.test_images = _count_images(test_dir)

    stats.names = names
    stats.classes = len(names) if names else (nc or 0)

    # Sample readability checks
    for directory in [train_dir, val_dir]:
        if not directory or not directory.exists():
            continue
        for image_path in _sample_images(directory, limit=3):
            if not _check_image_readable(image_path):
                messages.append(
                    ValidationMessage(
                        level="error", text=f"Unreadable image: {image_path}"
                    )
                )

    # Label sanity (best effort)
    labels_dir = _resolve(config.labels_dir, root) if config.labels_dir else None
    label_checks: List[Path] = []
    if config.task_type != "classify":
        if labels_dir and labels_dir.exists():
            label_checks = [
                p
                for p in labels_dir.rglob("*.txt")
                if p.is_file() and not p.name.startswith(".")
            ][:5]
        for label_path in label_checks:
            if not _validate_label_file(label_path, config.task_type):
                messages.append(
                    ValidationMessage(
                        level="warning",
                        text=f"Label format issue: {label_path.name}",
                    )
                )
                break

    ok = not any(m.level == "error" for m in messages)
    return DatasetValidationResult(ok=ok, messages=messages, stats=stats)
