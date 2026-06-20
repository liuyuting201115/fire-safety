#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_hazard.config import load_config, repository_path
from fire_hazard.data import evaluation_transform
from fire_hazard.evaluation import select_device
from fire_hazard.models import build_model, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one image")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.device:
        config["device"] = args.device
    device = select_device(config.get("device", "auto"))

    model = build_model(config)
    load_checkpoint(model, repository_path(config["checkpoint"]))
    model.to(device).eval()
    image = Image.open(args.image).convert("RGB")
    tensor = evaluation_transform(config)(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        if config["task"] == "binary":
            positive = torch.sigmoid(logits).item()
            probabilities = [1.0 - positive, positive]
        else:
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    result = {
        "image": str(Path(args.image)),
        "predicted_class": config["data"]["class_names"][index],
        "confidence": probabilities[index],
        "probabilities": dict(zip(config["data"]["class_names"], probabilities)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
