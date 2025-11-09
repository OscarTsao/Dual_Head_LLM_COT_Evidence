#!/usr/bin/env python3
"""
Prepare redsm5 dataset for evidence binding with DHRD.

This script:
1. Loads annotations.csv with status=1 filter
2. Splits data into train/val/test
3. Optionally generates rationales for evidence
4. Saves processed data ready for DHRD training

Usage:
    python scripts/prepare_redsm5.py --input data/redsm5/annotations.csv --output data/redsm5
    python scripts/prepare_redsm5.py --input data/redsm5/annotations.csv --output data/redsm5 --generate-rationales
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.SubProject.data.evidence_binding_dataset import prepare_redsm5_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare redsm5 dataset for evidence binding"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to annotations.csv file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--status-filter",
        type=int,
        default=1,
        help="Filter rows by status value (default: 1)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio for training set (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio for test set (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--generate-rationales",
        action="store_true",
        help="Generate rationales for evidence (requires teacher model)"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show dataset statistics"
    )

    return parser.parse_args()


def show_statistics(df: pd.DataFrame, name: str = "Dataset"):
    """Show dataset statistics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"\nColumn information:")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"\nStatus distribution:")
    if 'status' in df.columns:
        logger.info(f"{df['status'].value_counts().to_dict()}")
    logger.info(f"\nMissing values:")
    logger.info(f"{df.isnull().sum().to_dict()}")

    # Show sample rows
    logger.info(f"\nSample rows (first 3):")
    logger.info(f"\n{df.head(3).to_string()}")
    logger.info(f"{'='*60}\n")


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    """Validate that ratios sum to 1.0."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


def main():
    """Main function."""
    args = parse_args()

    logger.info("Starting redsm5 data preparation")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.info("\nPlease ensure your annotations.csv file is located at:")
        logger.info(f"  {args.input}")
        logger.info("\nExpected format:")
        logger.info("  - CSV file with 'status' column for filtering")
        logger.info("  - 'sentence' column containing evidence ground truth")
        logger.info("  - Other columns for context/input text")
        return

    # Validate ratios
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    # Show input statistics if requested
    if args.show_stats:
        df_full = pd.read_csv(input_path)
        show_statistics(df_full, "Input Data")

    # Prepare data splits
    logger.info("Splitting data into train/val/test...")
    train_df, val_df, test_df = prepare_redsm5_data(
        annotations_csv=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        status_filter=args.status_filter,
        random_seed=args.seed
    )

    # Show split statistics
    if args.show_stats:
        show_statistics(train_df, "Training Set")
        show_statistics(val_df, "Validation Set")
        show_statistics(test_df, "Test Set")

    # Generate rationales if requested
    if args.generate_rationales:
        logger.info("\nNote: Rationale generation requires a teacher model.")
        logger.info("Rationales will be auto-generated during training based on evidence sentences.")
        logger.info("For custom rationales, use scripts/generate_rationales.py on the split files.")

    logger.info("\n" + "="*60)
    logger.info("Data preparation completed successfully!")
    logger.info("="*60)
    logger.info(f"\nOutput files:")
    logger.info(f"  Train: {Path(args.output) / 'train_annotations.csv'}")
    logger.info(f"  Val:   {Path(args.output) / 'val_annotations.csv'}")
    logger.info(f"  Test:  {Path(args.output) / 'test_annotations.csv'}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Review the split data in {args.output}")
    logger.info(f"  2. Train DHRD model with:")
    logger.info(f"     python scripts/train_evidence_binding.py \\")
    logger.info(f"       --data_dir {args.output} \\")
    logger.info(f"       --model gpt2 \\")
    logger.info(f"       --output_dir outputs/evidence_binding")
    logger.info("")


if __name__ == "__main__":
    main()
