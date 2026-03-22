#!/usr/bin/env python3
"""Run a Spiritualist OCR inference experiment from a YAML config."""

import os
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")        # expose GPU to Paddle Inference C++ API
os.environ["FLAGS_use_mkldnn"] = "0"                       # disable oneDNN — PIR bug in PaddlePaddle 3.x
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # skip slow connectivity check

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
