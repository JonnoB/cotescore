#!/usr/bin/env python
"""
Script for downloading the NCSE v2 dataset.

Dataset URL: https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610
"""

import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download NCSE v2 dataset")
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Output directory")

    args = parser.parse_args()

    logger.info("NCSE v2 Dataset Download")
    logger.info("=" * 50)
    logger.info(
        "Dataset URL: https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610"
    )
    logger.info("")
    logger.info("Please download the dataset manually from the URL above and extract it to:")
    logger.info(f"  {args.output.absolute()}")
    logger.info("")
    logger.info("TODO: Implement automated download if API is available")


if __name__ == "__main__":
    main()
