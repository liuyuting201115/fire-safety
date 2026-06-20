from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class TargetMap:
    def __init__(self, mapping: dict[int, int]):
        self.mapping = dict(mapping)

    def __call__(self, target: int) -> int:
        return self.mapping[target]


def evaluation_transform(config: dict):
    data_config = config["data"]
    operations = [
        transforms.Resize(
            (config["model"]["image_size"], config["model"]["image_size"]),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ]
    mean = data_config.get("normalize_mean")
    std = data_config.get("normalize_std")
    if mean is not None and std is not None:
        operations.append(transforms.Normalize(mean, std))
    return transforms.Compose(operations)


def build_evaluation_dataset(data_dir: str | Path, config: dict):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Test dataset not found: {data_dir}")

    dataset = datasets.ImageFolder(data_dir, transform=evaluation_transform(config))
    class_names = list(config["data"]["class_names"])
    missing = [name for name in class_names if name not in dataset.class_to_idx]
    extra = [name for name in dataset.classes if name not in class_names]
    if missing or extra:
        raise ValueError(
            f"Dataset classes do not match config. missing={missing}, extra={extra}, "
            f"found={dataset.classes}"
        )

    mapping = {
        dataset.class_to_idx[name]: new_index
        for new_index, name in enumerate(class_names)
    }
    dataset.target_transform = TargetMap(mapping)
    return dataset
