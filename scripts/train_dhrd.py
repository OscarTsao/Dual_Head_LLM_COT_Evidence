#!/usr/bin/env python3
"""
Example training script for Dual-Head Reasoning Distillation (DHRD) model.

This script demonstrates how to train a DHRD model on classification tasks
with teacher-generated rationales.

Usage:
    python scripts/train_dhrd.py --config configs/dhrd_config.yaml
    python scripts/train_dhrd.py --model gpt2 --num_labels 3 --data_path data/train.json
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
from Project.SubProject.data.dhrd_dataset import DHRDDataset, create_dhrd_dataloader
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
        description="Train DHRD model for classification with reasoning distillation"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Pretrained model name or path (e.g., 'gpt2', 'facebook/opt-125m')"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        required=True,
        help="Number of classification labels"
    )
    parser.add_argument(
        "--reasoning_weight",
        type=float,
        default=0.5,
        help="Weight for reasoning loss in combined loss (lambda)"
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
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSON/JSONL)"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        help="Path to validation data (JSON/JSONL)"
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
        help="Maximum sequence length for input+rationale"
    )
    parser.add_argument(
        "--include_rationales",
        action="store_true",
        default=True,
        help="Include rationales in training"
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
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
        help="Maximum gradient norm for clipping"
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
        default="outputs/dhrd",
        help="Directory to save checkpoints and logs"
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
        help="Use MLflow for logging"
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="dhrd_training",
        help="MLflow experiment name"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )

    return parser.parse_args()


def load_data(args):
    """Load and prepare datasets."""
    logger = get_logger(__name__)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    logger.info(f"Loading training data from: {args.train_data}")
    train_dataset = DHRDDataset(
        data=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_rationale_length=args.max_rationale_length,
        include_rationales=args.include_rationales,
    )

    train_dataloader = create_dhrd_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")

    # Load validation data
    val_dataloader = None
    if args.val_data:
        logger.info(f"Loading validation data from: {args.val_data}")
        val_dataset = DHRDDataset(
            data=args.val_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_rationale_length=args.max_rationale_length,
            include_rationales=False,  # No rationales needed for validation
        )

        val_dataloader = create_dhrd_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(f"Validation dataset size: {len(val_dataset)}")

    return train_dataloader, val_dataloader, tokenizer


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logger = get_logger(__name__)
    logger.info("Starting DHRD training")
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
    train_dataloader, val_dataloader, tokenizer = load_data(args)

    # Create model
    logger.info(f"Creating DHRD model with backbone: {args.model}")
    model = DHRDModel(
        backbone_model_name=args.model,
        num_labels=args.num_labels,
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
        configure_mlflow(
            tracking_uri="file:./mlruns",
            experiment=args.mlflow_experiment
        )

        with mlflow_run("dhrd_training", tags={"model": args.model}):
            # Log parameters
            mlflow.log_params({
                "model": args.model,
                "num_labels": args.num_labels,
                "reasoning_weight": args.reasoning_weight,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
            })

            # Train
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
        final_metrics = trainer.train(num_epochs=args.num_epochs)

    logger.info(f"Training completed!")
    logger.info(f"Final metrics: {final_metrics}")
    logger.info(f"Best validation accuracy: {final_metrics['best_val_accuracy']:.4f}")

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir / "tokenizer")
    logger.info(f"Tokenizer saved to: {output_dir / 'tokenizer'}")


if __name__ == "__main__":
    main()
