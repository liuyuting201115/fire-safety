#!/usr/bin/env python3
"""Create deterministic 70/15/15 train, validation, and test splits.

The source directory must contain the two legacy datasets used in the project:

    fire_image/train|val/<class>/*
    fire_image_binary/train|val/<class>/*

Outputs follow this repository's private data layout under data/train and
data/test. Images are never intended to be committed to the public repository.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASETS = {
    "multiclass": {
        "source": "fire_image",
        "classes": ["Confused_Wiring", "Fire_Lane_Blocked"],
    },
    "binary": {
        "source": "fire_image_binary",
        "classes": ["Hazard", "No_Hazard"],
    },
}


def collect_images(dataset_dir: Path, class_names: list[str]) -> dict[str, list[Path]]:
    pools: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "val"):
        for class_name in class_names:
            class_dir = dataset_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing source directory: {class_dir}")
            pools[class_name].extend(
                path
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return pools


def stratified_split(
    paths: list[Path],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[Path]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count == 0:
        return {"train": [], "val": [], "test": []}
    test_count = max(1, round(count * test_ratio)) if count >= 3 else 0
    val_count = max(1, round(count * val_ratio)) if count >= 3 else 0
    train_count = count - val_count - test_count
    if train_count < 1:
        train_count = 1
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def reset_output_directories(output_root: Path, overwrite: bool) -> None:
    targets = [output_root / "train", output_root / "test", output_root / "private_manifests"]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {joined}. Pass --overwrite to replace it.")
    for path in existing:
        shutil.rmtree(path)
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)


def prepare_dataset(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    reset_output_directories(output_root, args.overwrite)
    summary_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for task, specification in DATASETS.items():
        class_names = specification["classes"]
        pools = collect_images(source_root / specification["source"], class_names)
        train_task_root = output_root / "train" / task
        test_task_root = output_root / "test" / task
        (train_task_root / "class_names.txt").parent.mkdir(parents=True, exist_ok=True)
        (train_task_root / "class_names.txt").write_text(
            "\n".join(class_names) + "\n", encoding="utf-8"
        )

        for class_name in class_names:
            parts = stratified_split(
                pools[class_name],
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )
            summary_rows.append(
                {
                    "task": task,
                    "class": class_name,
                    "train": len(parts["train"]),
                    "val": len(parts["val"]),
                    "test": len(parts["test"]),
                    "total": len(pools[class_name]),
                }
            )
            for split, sources in parts.items():
                destination_dir = (
                    test_task_root / class_name
                    if split == "test"
                    else train_task_root / split / class_name
                )
                destination_dir.mkdir(parents=True, exist_ok=True)
                for index, source in enumerate(sources, start=1):
                    destination = destination_dir / (
                        f"{class_name}_{split}_{index:04d}{source.suffix.lower()}"
                    )
                    shutil.copy2(source, destination)
                    manifest_rows.append(
                        {
                            "task": task,
                            "split": split,
                            "class": class_name,
                            "new_path": destination.relative_to(output_root).as_posix(),
                            "original_path": source.relative_to(source_root).as_posix(),
                            "original_filename": source.name,
                        }
                    )

    private_manifest_root = output_root / "private_manifests"
    with (private_manifest_root / "dataset_split_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["task", "class", "train", "val", "test", "total"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with (private_manifest_root / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task",
                "split",
                "class",
                "new_path",
                "original_path",
                "original_filename",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Dataset split completed")
    for row in summary_rows:
        print(row)
    print(f"Private manifests: {private_manifest_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_dataset(parse_args())
