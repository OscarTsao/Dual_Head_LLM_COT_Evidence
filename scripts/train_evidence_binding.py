#!/usr/bin/env python3
"""
Training script for Evidence Binding with DHRD.

This script trains a DHRD model specifically for evidence binding tasks
using the redsm5 dataset with status=1 annotations.

Usage:
    python scripts/train_evidence_binding.py \
        --data_dir data/redsm5 \
        --model gpt2 \
        --output_dir outputs/evidence_binding
"""

import argparse
import sys
import logging
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.SubProject.models.dhrd_model import DHRDModel
from Project.SubProject.data.evidence_binding_dataset import (
    EvidenceBindingDataset,
    EvidenceBindingDataModule
)
from Project.SubProject.data.dhrd_dataset import create_dhrd_dataloader
from Project.SubProject.engine.dhrd_trainer import (
    DHRDTrainer,
    create_optimizer,
    create_scheduler
)
from Project.SubProject.utils.seed import set_seed
from Project.SubProject.utils.log import get_logger

try:
    import mlflow
    from Project.SubProject.utils.mlflow_utils import configure_mlflow, mlflow_run
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DHRD model for evidence binding"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Pretrained model name (e.g., 'gpt2', 'facebook/opt-125m')"
    )
    parser.add_argument(
        "--reasoning_weight",
        type=float,
        default=0.5,
        help="Weight for reasoning loss (lambda)"
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Pooling method for classification head"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train/val/test CSV files"
    )
    parser.add_argument(
        "--status_filter",
        type=int,
        default=1,
        help="Filter by status value (default: 1)"
    )
    parser.add_argument(
        "--evidence_column",
        type=str,
        default="sentence",
        help="Column name for evidence ground truth"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for input"
    )
    parser.add_argument(
        "--max_rationale_length",
        type=int,
        default=1024,
        help="Maximum length for input+rationale"
    )
    parser.add_argument(
        "--include_rationales",
        action="store_true",
        default=True,
        help="Include rationales in training"
    )
    parser.add_argument(
        "--binary_classification",
        action="store_true",
        default=True,
        help="Use binary classification (has evidence: 0/1)"
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Logging and saving
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evidence_binding",
        help="Output directory"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Steps between logging"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=100,
        help="Steps between evaluation"
    )
    parser.add_argument(
        "--use_mlflow",
        action="store_true",
        help="Use MLflow logging"
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="evidence_binding",
        help="MLflow experiment name"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    return parser.parse_args()


def load_evidence_binding_data(args):
    """Load evidence binding datasets."""
    logger = get_logger(__name__)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup data module
    data_dir = Path(args.data_dir)

    # Check if split files exist
    train_file = data_dir / "train_annotations.csv"
    val_file = data_dir / "val_annotations.csv"

    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.info("\nPlease run the data preparation script first:")
        logger.info(f"  python scripts/prepare_redsm5.py \\")
        logger.info(f"    --input data/redsm5/annotations.csv \\")
        logger.info(f"    --output {args.data_dir}")
        raise FileNotFoundError(f"Training file not found: {train_file}")

    # Create datasets
    logger.info(f"Loading training data from {train_file}")
    train_dataset = EvidenceBindingDataset(
        csv_path=train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_rationale_length=args.max_rationale_length,
        status_filter=None,  # Already filtered during preparation
        evidence_column=args.evidence_column,
        include_rationales=args.include_rationales,
        binary_classification=args.binary_classification,
    )

    # Show dataset statistics
    stats = train_dataset.get_statistics()
    logger.info(f"Training dataset statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Label distribution: {stats['label_distribution']}")
    logger.info(f"  Evidence present: {stats['evidence_present']}")
    logger.info(f"  Text length (mean): {stats['text_length']['mean']:.1f}")

    train_dataloader = create_dhrd_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Load validation data if available
    val_dataloader = None
    if val_file.exists():
        logger.info(f"Loading validation data from {val_file}")
        val_dataset = EvidenceBindingDataset(
            csv_path=val_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_rationale_length=args.max_rationale_length,
            status_filter=None,
            evidence_column=args.evidence_column,
            include_rationales=False,  # No rationales needed for validation
            binary_classification=args.binary_classification,
        )

        val_stats = val_dataset.get_statistics()
        logger.info(f"Validation dataset statistics:")
        logger.info(f"  Total samples: {val_stats['total_samples']}")
        logger.info(f"  Label distribution: {val_stats['label_distribution']}")

        val_dataloader = create_dhrd_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    return train_dataloader, val_dataloader, tokenizer, train_dataset


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logger = get_logger(__name__)
    logger.info("="*60)
    logger.info("Evidence Binding with DHRD Training")
    logger.info("="*60)
    logger.info(f"Arguments: {vars(args)}")

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    logger.info("\nLoading data...")
    train_dataloader, val_dataloader, tokenizer, train_dataset = load_evidence_binding_data(args)

    # Determine number of labels
    if args.binary_classification:
        num_labels = 2
    else:
        # Auto-detect from data
        num_labels = len(train_dataset.df[train_dataset.label_column].unique())

    logger.info(f"\nNumber of labels: {num_labels}")

    # Create model
    logger.info(f"\nCreating DHRD model with backbone: {args.model}")
    model = DHRDModel(
        backbone_model_name=args.model,
        num_labels=num_labels,
        reasoning_weight=args.reasoning_weight,
        pooling_method=args.pooling_method,
        use_pretrained=True,
    )

    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(f"Trainable parameters: {model.get_num_parameters(only_trainable=True):,}")

    # Create optimizer and scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = create_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=args.warmup_ratio,
        scheduler_type="linear",
    )

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = DHRDTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=args.device,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_dir=str(output_dir),
        use_mlflow=args.use_mlflow,
    )

    # Setup MLflow
    if args.use_mlflow and MLFLOW_AVAILABLE:
        logger.info("\nSetting up MLflow...")
        configure_mlflow(
            tracking_uri="file:./mlruns",
            experiment=args.mlflow_experiment
        )

        with mlflow_run("evidence_binding", tags={"model": args.model, "task": "evidence_binding"}):
            # Log parameters
            mlflow.log_params({
                "model": args.model,
                "num_labels": num_labels,
                "reasoning_weight": args.reasoning_weight,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "evidence_column": args.evidence_column,
            })

            # Train
            logger.info("\n" + "="*60)
            logger.info("Starting training...")
            logger.info("="*60 + "\n")
            final_metrics = trainer.train(num_epochs=args.num_epochs)

            # Log final metrics
            mlflow.log_metrics({
                "final/best_val_accuracy": final_metrics["best_val_accuracy"],
                "final/total_steps": final_metrics["total_steps"],
            })

            # Log model
            mlflow.pytorch.log_model(model, "model")

    else:
        # Train without MLflow
        logger.info("\n" + "="*60)
        logger.info("Starting training...")
        logger.info("="*60 + "\n")
        final_metrics = trainer.train(num_epochs=args.num_epochs)

    # Training complete
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)
    logger.info(f"\nFinal metrics:")
    logger.info(f"  Best validation accuracy: {final_metrics['best_val_accuracy']:.4f}")
    logger.info(f"  Total training steps: {final_metrics['total_steps']}")

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"\nFinal model saved to: {final_model_path}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir / "tokenizer")
    logger.info(f"Tokenizer saved to: {output_dir / 'tokenizer'}")

    logger.info("\nModel is ready for inference!")
    logger.info(f"Use: python scripts/inference_dhrd.py --model_path {output_dir}")


if __name__ == "__main__":
    main()
