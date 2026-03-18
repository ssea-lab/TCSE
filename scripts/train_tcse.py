#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcse import TCSETrainer, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCSE")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    trainer = TCSETrainer(cfg)
    if not args.eval_only:
        trainer.train()
    metrics = trainer.evaluate(split="test")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
