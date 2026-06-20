#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_hazard.config import load_config
from fire_hazard.evaluation import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fire-hazard checkpoint")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--checkpoint", help="Override checkpoint path")
    parser.add_argument("--data-dir", help="Override test dataset path")
    parser.add_argument("--output-dir", help="Override result directory")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--no-verify", action="store_true", help="Do not fail on metric drift")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.data_dir:
        config["data"]["test_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.device:
        config["device"] = args.device

    metrics, passed = evaluate(config, verify_expected=not args.no_verify)
    summary = metrics["metrics"]
    print(f"Task: {metrics['task']}")
    print(f"Samples: {metrics['dataset']['samples']}")
    print(f"Accuracy: {summary['accuracy']:.6f}")
    print(f"Macro F1: {summary['macro_f1']:.6f}")
    print(f"Result directory: {config['output_dir']}")
    print(f"Reference verification: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
