"""
Trainer for Dual-Head Reasoning Distillation (DHRD) models.

This module provides training utilities for DHRD models with support for:
- Dual-head training with classification and reasoning losses
- Teacher rationale distillation
- MLflow logging
- Evaluation and metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..models.dhrd_model import DHRDModel


logger = logging.getLogger(__name__)


class DHRDTrainer:
    """
    Trainer class for Dual-Head Reasoning Distillation models.
    """

    def __init__(
        self,
        model: DHRDModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_dir: Optional[str] = None,
        use_mlflow: bool = False,
        **kwargs
    ):
        """
        Args:
            model: DHRDModel instance
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            optimizer: Optimizer instance (if None, AdamW will be created)
            lr_scheduler: Learning rate scheduler
            device: Device to use for training
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            save_dir: Directory to save checkpoints
            use_mlflow: Whether to log to MLflow
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = Path(save_dir) if save_dir else None
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=kwargs.get('learning_rate', 2e-5),
                weight_decay=kwargs.get('weight_decay', 0.01)
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_accuracy = 0.0

        # Create save directory
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.model.enable_reasoning_head()

        total_loss = 0.0
        total_classification_loss = 0.0
        total_reasoning_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            leave=True
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                rationale_input_ids=batch.get("rationale_input_ids"),
                rationale_attention_mask=batch.get("rationale_attention_mask"),
                rationale_labels=batch.get("rationale_labels"),
            )

            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Update metrics
            total_loss += loss.item()
            if outputs.classification_loss is not None:
                total_classification_loss += outputs.classification_loss.item()
            if outputs.reasoning_loss is not None:
                total_reasoning_loss += outputs.reasoning_loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "cls_loss": outputs.classification_loss.item() if outputs.classification_loss else 0.0,
                "rsn_loss": outputs.reasoning_loss.item() if outputs.reasoning_loss else 0.0,
            })

            # Log to MLflow
            if self.use_mlflow and self.global_step % self.log_interval == 0:
                mlflow.log_metrics({
                    "train/loss": loss.item(),
                    "train/classification_loss": outputs.classification_loss.item() if outputs.classification_loss else 0.0,
                    "train/reasoning_loss": outputs.reasoning_loss.item() if outputs.reasoning_loss else 0.0,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                }, step=self.global_step)

            # Evaluate
            if self.val_dataloader and self.global_step % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Step {self.global_step}: {eval_metrics}")

                if self.use_mlflow:
                    mlflow.log_metrics({
                        f"val/{k}": v for k, v in eval_metrics.items()
                    }, step=self.global_step)

                # Save best model
                if eval_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = eval_metrics['accuracy']
                    if self.save_dir:
                        self.save_checkpoint('best_model.pt')

                self.model.train()
                self.model.enable_reasoning_head()

        # Calculate epoch metrics
        metrics = {
            "loss": total_loss / num_batches,
            "classification_loss": total_classification_loss / num_batches,
            "reasoning_loss": total_reasoning_loss / num_batches,
        }

        return metrics

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Dictionary of final metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total steps: {len(self.train_dataloader) * num_epochs}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = self.train_epoch()

            logger.info(f"Epoch {epoch + 1}/{num_epochs} - {epoch_metrics}")

            # Evaluate at end of epoch
            if self.val_dataloader:
                eval_metrics = self.evaluate()
                logger.info(f"Validation: {eval_metrics}")

                if self.use_mlflow:
                    mlflow.log_metrics({
                        f"epoch_val/{k}": v for k, v in eval_metrics.items()
                    }, step=epoch)

            # Save checkpoint
            if self.save_dir:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        logger.info("Training completed!")

        return {
            "best_val_accuracy": self.best_val_accuracy,
            "final_epoch": self.epoch,
            "total_steps": self.global_step,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        self.model.disable_reasoning_head()  # Use fast inference mode

        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass (classification only)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            # Get predictions
            predictions = torch.argmax(outputs.classification_logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            if outputs.classification_loss is not None:
                total_loss += outputs.classification_loss.item()
            num_batches += 1

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = (all_predictions == all_labels).mean()
        loss = total_loss / num_batches if num_batches > 0 else 0.0

        metrics = {
            "accuracy": float(accuracy),
            "loss": float(loss),
        }

        return metrics

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        if self.save_dir is None:
            logger.warning("save_dir not set, skipping checkpoint save")
            return

        checkpoint_path = self.save_dir / filename

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_accuracy = checkpoint['best_val_accuracy']

        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with weight decay.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        adam_epsilon: Epsilon for Adam
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    # Separate parameters that should/shouldn't have weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon,
        **kwargs
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "linear",
) -> Any:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        warmup_ratio: Ratio of warmup steps to total steps
        scheduler_type: Type of scheduler ('linear', 'cosine', etc.)

    Returns:
        Scheduler instance
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math

    if num_warmup_steps is None:
        num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        if scheduler_type == "linear":
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )
        elif scheduler_type == "cosine":
            progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
