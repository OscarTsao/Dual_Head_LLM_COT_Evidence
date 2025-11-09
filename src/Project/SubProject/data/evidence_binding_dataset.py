"""
Evidence Binding Dataset for DHRD Training

This module provides dataset utilities for evidence binding tasks using
the redsm5 dataset with DHRD (Dual-Head Reasoning Distillation).

Evidence binding involves:
- Input: Medical case or document text
- Task: Identify and extract relevant evidence sentences
- Ground Truth: Evidence sentences from annotations
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EvidenceBindingDataset(Dataset):
    """
    Dataset for evidence binding task using annotations.csv from redsm5.

    Expects annotations.csv with columns:
    - status: Filter for status=1 (valid cases)
    - sentence: Evidence ground truth sentences
    - Other contextual columns for building input text
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_rationale_length: int = 1024,
        status_filter: int = 1,
        text_columns: Optional[List[str]] = None,
        evidence_column: str = "sentence",
        label_column: Optional[str] = None,
        rationale_column: Optional[str] = "rationale",
        include_rationales: bool = True,
        binary_classification: bool = True,
        **kwargs
    ):
        """
        Args:
            csv_path: Path to annotations.csv file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length for input
            max_rationale_length: Maximum length for input+rationale
            status_filter: Filter rows by status value (default: 1)
            text_columns: List of column names to concatenate for input text
                         If None, uses all columns except evidence and metadata
            evidence_column: Column name containing evidence sentences
            label_column: Column name for labels (if None, auto-generates binary labels)
            rationale_column: Column name for rationales (if available)
            include_rationales: Whether to include rationales in training
            binary_classification: If True, creates binary labels (has_evidence: 0/1)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_rationale_length = max_rationale_length
        self.evidence_column = evidence_column
        self.label_column = label_column
        self.rationale_column = rationale_column
        self.include_rationales = include_rationales
        self.binary_classification = binary_classification

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and filter data
        logger.info(f"Loading data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} total rows")

        # Filter by status
        if status_filter is not None:
            self.df = self.df[self.df['status'] == status_filter]
            logger.info(f"Filtered to {len(self.df)} rows with status={status_filter}")

        if len(self.df) == 0:
            raise ValueError(f"No data remaining after filtering for status={status_filter}")

        # Determine text columns
        if text_columns is None:
            # Auto-detect: use all columns except evidence, labels, and metadata
            exclude_cols = {evidence_column, label_column, rationale_column, 'status', 'id', 'index'}
            text_columns = [col for col in self.df.columns if col not in exclude_cols]
            logger.info(f"Auto-detected text columns: {text_columns}")

        self.text_columns = text_columns

        # Create labels if not provided
        if self.label_column is None or self.label_column not in self.df.columns:
            if self.binary_classification:
                # Binary: 1 if evidence exists, 0 otherwise
                self.df['label'] = self.df[evidence_column].notna().astype(int)
                self.label_column = 'label'
                logger.info("Created binary labels based on evidence presence")
            else:
                raise ValueError("label_column must be specified if not using binary_classification")

        # Reset index
        self.df = self.df.reset_index(drop=True)

        logger.info(f"Dataset initialized with {len(self.df)} samples")
        logger.info(f"Label distribution: {self.df[self.label_column].value_counts().to_dict()}")

    def __len__(self) -> int:
        return len(self.df)

    def _build_input_text(self, row: pd.Series) -> str:
        """
        Build input text from row data.

        Args:
            row: DataFrame row

        Returns:
            Formatted input text
        """
        parts = []
        for col in self.text_columns:
            if col in row and pd.notna(row[col]):
                # Format as "Column: Value"
                col_name = col.replace('_', ' ').title()
                parts.append(f"{col_name}: {row[col]}")

        return "\n".join(parts)

    def _build_rationale_text(self, row: pd.Series, input_text: str) -> str:
        """
        Build rationale text including evidence.

        Args:
            row: DataFrame row
            input_text: Input text already built

        Returns:
            Rationale text with reasoning about evidence
        """
        # Check if explicit rationale column exists
        if self.rationale_column in row and pd.notna(row[self.rationale_column]):
            return str(row[self.rationale_column])

        # Auto-generate rationale from evidence
        evidence = row[self.evidence_column]
        label = row[self.label_column]

        if pd.notna(evidence) and label == 1:
            rationale = f"""Let me analyze this step by step to identify the key evidence:

1. Reviewing the provided information carefully
2. Identifying relevant clinical/factual details
3. Extracting the key evidence sentence

The key evidence is: "{evidence}"

This evidence is relevant because it directly supports the claim and provides specific factual information."""
        else:
            rationale = f"""Let me analyze this step by step:

1. Reviewing the provided information
2. Looking for supporting evidence
3. Evaluating the available data

Based on my analysis, there is no clear evidence sentence that directly supports this claim in the provided information."""

        return rationale

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Returns:
            Dictionary containing:
                - input_ids: Tokenized input text
                - attention_mask: Attention mask for input
                - labels: Classification label
                - rationale_input_ids: (optional) Tokenized input+rationale
                - rationale_attention_mask: (optional) Attention mask for rationale
                - rationale_labels: (optional) Labels for LM loss
                - evidence_text: (optional) Original evidence text for reference
        """
        row = self.df.iloc[idx]

        # Build input text
        input_text = self._build_input_text(row)

        # Get label
        label = int(row[self.label_column])

        # Tokenize input
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        # Add rationale if requested
        if self.include_rationales:
            rationale = self._build_rationale_text(row, input_text)

            # Concatenate input and rationale
            combined_text = f"{input_text}\n\n{rationale}"

            rationale_encoding = self.tokenizer(
                combined_text,
                max_length=self.max_rationale_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            result["rationale_input_ids"] = rationale_encoding["input_ids"].squeeze(0)
            result["rationale_attention_mask"] = rationale_encoding["attention_mask"].squeeze(0)

            # For LM loss, labels are the same as input_ids
            # Set padding tokens to -100 to ignore in loss
            rationale_labels = rationale_encoding["input_ids"].squeeze(0).clone()
            rationale_labels[rationale_labels == self.tokenizer.pad_token_id] = -100
            result["rationale_labels"] = rationale_labels

        # Add evidence text for reference (not used in training)
        if self.evidence_column in row:
            evidence = row[self.evidence_column]
            result["evidence_text"] = str(evidence) if pd.notna(evidence) else ""

        return result

    def get_evidence_texts(self) -> List[str]:
        """
        Get all evidence texts from the dataset.

        Returns:
            List of evidence texts
        """
        return [
            str(e) if pd.notna(e) else ""
            for e in self.df[self.evidence_column]
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": len(self.df),
            "label_distribution": self.df[self.label_column].value_counts().to_dict(),
            "evidence_present": self.df[self.evidence_column].notna().sum(),
            "evidence_missing": self.df[self.evidence_column].isna().sum(),
        }

        # Add text length statistics
        text_lengths = [
            len(self._build_input_text(row))
            for _, row in self.df.iterrows()
        ]

        stats["text_length"] = {
            "mean": sum(text_lengths) / len(text_lengths),
            "min": min(text_lengths),
            "max": max(text_lengths),
        }

        return stats


class EvidenceBindingDataModule:
    """
    Data module for managing train/val/test splits for evidence binding.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        train_file: str = "train_annotations.csv",
        val_file: str = "val_annotations.csv",
        test_file: str = "test_annotations.csv",
        **dataset_kwargs
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            tokenizer: Tokenizer for encoding
            train_file: Name of training CSV file
            val_file: Name of validation CSV file
            test_file: Name of test CSV file
            **dataset_kwargs: Additional arguments for EvidenceBindingDataset
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.dataset_kwargs = dataset_kwargs

        self.train_path = self.data_dir / train_file
        self.val_path = self.data_dir / val_file
        self.test_path = self.data_dir / test_file

    def setup(self) -> Tuple[EvidenceBindingDataset, ...]:
        """
        Setup train/val/test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        datasets = []

        for path, name in [
            (self.train_path, "train"),
            (self.val_path, "val"),
            (self.test_path, "test")
        ]:
            if path.exists():
                logger.info(f"Loading {name} dataset from {path}")
                dataset = EvidenceBindingDataset(
                    csv_path=path,
                    tokenizer=self.tokenizer,
                    **self.dataset_kwargs
                )
                datasets.append(dataset)
            else:
                logger.warning(f"{name} dataset not found at {path}")
                datasets.append(None)

        return tuple(datasets)


def prepare_redsm5_data(
    annotations_csv: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    status_filter: int = 1,
    random_seed: int = 42
):
    """
    Prepare redsm5 data by splitting annotations.csv into train/val/test.

    Args:
        annotations_csv: Path to annotations.csv
        output_dir: Output directory for split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        status_filter: Filter by status value
        random_seed: Random seed for reproducibility
    """
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {annotations_csv}")
    df = pd.read_csv(annotations_csv)
    logger.info(f"Loaded {len(df)} rows")

    # Filter by status
    if status_filter is not None:
        df = df[df['status'] == status_filter]
        logger.info(f"Filtered to {len(df)} rows with status={status_filter}")

    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Save splits
    train_df.to_csv(output_dir / "train_annotations.csv", index=False)
    val_df.to_csv(output_dir / "val_annotations.csv", index=False)
    test_df.to_csv(output_dir / "test_annotations.csv", index=False)

    logger.info(f"Saved splits:")
    logger.info(f"  Train: {len(train_df)} samples -> {output_dir / 'train_annotations.csv'}")
    logger.info(f"  Val: {len(val_df)} samples -> {output_dir / 'val_annotations.csv'}")
    logger.info(f"  Test: {len(test_df)} samples -> {output_dir / 'test_annotations.csv'}")

    return train_df, val_df, test_df


def create_dhrd_dataloader(
    dataset: EvidenceBindingDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for DHRD training.

    Args:
        dataset: EvidenceBindingDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
