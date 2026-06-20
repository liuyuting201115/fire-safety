from __future__ import annotations

import csv
import json
import platform
import random
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import timm
import torch
import torchvision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import repository_path
from .data import build_evaluation_dataset
from .models import build_model, load_checkpoint


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def _write_predictions(
    path: Path,
    relative_paths: list[str],
    targets: list[int],
    predictions: list[int],
    confidence: list[float],
    class_names: list[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "true_label", "predicted_label", "confidence"])
        for sample_path, target, prediction, score in zip(
            relative_paths, targets, predictions, confidence
        ):
            writer.writerow(
                [sample_path, class_names[target], class_names[prediction], f"{score:.8f}"]
            )


@torch.inference_mode()
def evaluate(config: dict, verify_expected: bool = True) -> tuple[dict, bool]:
    set_seed(int(config.get("seed", 42)))
    device = select_device(config.get("device", "auto"))
    data_dir = repository_path(config["data"]["test_dir"])
    checkpoint = repository_path(config["checkpoint"])
    output_dir = repository_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_evaluation_dataset(data_dir, config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config)
    _, unexpected = load_checkpoint(model, checkpoint)
    model.to(device).eval()

    targets: list[int] = []
    predictions: list[int] = []
    confidence: list[float] = []
    for images, labels in loader:
        logits = model(images.to(device))
        if config["task"] == "binary":
            positive_probability = torch.sigmoid(logits).flatten()
            predicted = (positive_probability >= float(config.get("threshold", 0.5))).long()
            scores = torch.where(predicted == 1, positive_probability, 1 - positive_probability)
        else:
            probabilities = torch.softmax(logits, dim=1)
            scores, predicted = probabilities.max(dim=1)
        targets.extend(labels.tolist())
        predictions.extend(predicted.cpu().tolist())
        confidence.extend(scores.cpu().tolist())

    class_names = list(config["data"]["class_names"])
    labels = list(range(len(class_names)))
    accuracy = float(accuracy_score(targets, predictions))
    macro_f1 = float(f1_score(targets, predictions, labels=labels, average="macro"))
    report_text = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_data = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)

    expected = config.get("expected", {})
    tolerance = float(expected.get("tolerance", 1e-6))
    comparisons = {
        "accuracy": {
            "actual": accuracy,
            "expected": expected.get("accuracy"),
            "passed": expected.get("accuracy") is None
            or abs(accuracy - float(expected["accuracy"])) <= tolerance,
        },
        "macro_f1": {
            "actual": macro_f1,
            "expected": expected.get("macro_f1"),
            "passed": expected.get("macro_f1") is None
            or abs(macro_f1 - float(expected["macro_f1"])) <= tolerance,
        },
    }
    verified = all(item["passed"] for item in comparisons.values())

    class_counts = Counter(class_names[target] for target in targets)
    metrics = {
        "schema_version": 1,
        "task": config["task"],
        "dataset": {
            "path": str(data_dir.relative_to(repository_path("."))),
            "samples": len(dataset),
            "class_distribution": dict(class_counts),
        },
        "checkpoint": str(checkpoint.relative_to(repository_path("."))),
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "classification_report": report_data,
            "confusion_matrix": matrix.tolist(),
        },
        "verification": {
            "enabled": verify_expected,
            "passed": verified if verify_expected else None,
            "tolerance": tolerance,
            "comparisons": comparisons,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(device),
            "cuda_runtime": torch.version.cuda,
        },
        "checkpoint_load_notes": {
            "ignored_legacy_alias_keys": unexpected,
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    _write_confusion_matrix(output_dir / "confusion_matrix.csv", matrix, class_names)
    relative_paths = [
        Path(sample_path).relative_to(data_dir).as_posix()
        for sample_path, _ in dataset.samples
    ]
    _write_predictions(
        output_dir / "predictions.csv",
        relative_paths,
        targets,
        predictions,
        confidence,
        class_names,
    )
    return metrics, (verified or not verify_expected)
