#!/usr/bin/env python3
"""Run a Spiritualist OCR inference experiment from a YAML config."""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.config import load_config
from pipeline.runner import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run a Spiritualist OCR inference experiment")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment YAML config")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
