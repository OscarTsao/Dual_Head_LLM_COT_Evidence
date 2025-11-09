"""
Dataset classes for Dual-Head Reasoning Distillation (DHRD) training.

This module provides dataset utilities for loading and preprocessing data
with teacher-generated rationales for DHRD training.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path


class DHRDDataset(Dataset):
    """
    Dataset for DHRD training with classification labels and optional rationales.

    Supports:
    - Text classification with labels
    - Optional teacher-generated rationales (Chain-of-Thought)
    - Flexible input format (JSON, CSV, or list of dicts)
    """

    def __init__(
        self,
        data: Union[str, Path, List[Dict[str, Any]]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_rationale_length: int = 1024,
        text_column: str = "text",
        label_column: str = "label",
        rationale_column: str = "rationale",
        include_rationales: bool = True,
        rationale_format: str = "concat",  # 'concat' or 'separate'
    ):
        """
        Args:
            data: Path to data file (JSON/JSONL) or list of data dictionaries
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length for input text
            max_rationale_length: Maximum length for input+rationale
            text_column: Name of text column in data
            label_column: Name of label column in data
            rationale_column: Name of rationale column in data
            include_rationales: Whether to include rationales in dataset
            rationale_format: How to format rationales ('concat' or 'separate')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_rationale_length = max_rationale_length
        self.text_column = text_column
        self.label_column = label_column
        self.rationale_column = rationale_column
        self.include_rationales = include_rationales
        self.rationale_format = rationale_format

        # Load data
        self.data = self._load_data(data)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_data(
        self,
        data: Union[str, Path, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Load data from file or list."""
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if data_path.suffix == '.json':
                with open(data_path, 'r') as f:
                    return json.load(f)
            elif data_path.suffix == '.jsonl':
                items = []
                with open(data_path, 'r') as f:
                    for line in f:
                        items.append(json.loads(line))
                return items
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def __len__(self) -> int:
        return len(self.data)

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
        """
        item = self.data[idx]

        text = item[self.text_column]
        label = item[self.label_column]

        # Tokenize input text
        encoding = self.tokenizer(
            text,
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

        # Add rationale if available and requested
        if self.include_rationales and self.rationale_column in item:
            rationale = item[self.rationale_column]

            if self.rationale_format == "concat":
                # Concatenate input and rationale
                combined_text = f"{text}\n\n{rationale}"
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

            elif self.rationale_format == "separate":
                # Tokenize rationale separately
                rationale_encoding = self.tokenizer(
                    rationale,
                    max_length=self.max_rationale_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                result["rationale_input_ids"] = rationale_encoding["input_ids"].squeeze(0)
                result["rationale_attention_mask"] = rationale_encoding["attention_mask"].squeeze(0)

                rationale_labels = rationale_encoding["input_ids"].squeeze(0).clone()
                rationale_labels[rationale_labels == self.tokenizer.pad_token_id] = -100
                result["rationale_labels"] = rationale_labels

        return result


class SuperGLUEDataset(DHRDDataset):
    """
    Specialized dataset for SuperGLUE tasks.

    Supports tasks: CB, RTE, COPA, BoolQ, WiC, MultiRC, ReCoRD
    """

    TASK_FORMATS = {
        "cb": {"text": "premise", "text2": "hypothesis", "label": "label"},
        "rte": {"text": "premise", "text2": "hypothesis", "label": "label"},
        "copa": {"text": "premise", "choice1": "choice1", "choice2": "choice2", "question": "question", "label": "label"},
        "boolq": {"text": "passage", "question": "question", "label": "label"},
        "wic": {"text": "sentence1", "text2": "sentence2", "word": "word", "label": "label"},
        "multirc": {"text": "paragraph", "question": "question", "answer": "answer", "label": "label"},
    }

    def __init__(
        self,
        task_name: str,
        data: Union[str, Path, List[Dict[str, Any]]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_rationale_length: int = 1024,
        include_rationales: bool = True,
    ):
        """
        Args:
            task_name: Name of SuperGLUE task (e.g., 'cb', 'rte', 'copa')
            data: Path to data file or list of data dictionaries
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            max_rationale_length: Maximum length for input+rationale
            include_rationales: Whether to include rationales
        """
        self.task_name = task_name.lower()

        if self.task_name not in self.TASK_FORMATS:
            raise ValueError(f"Unsupported task: {task_name}")

        # Process data to create unified format
        processed_data = self._preprocess_task_data(data)

        super().__init__(
            data=processed_data,
            tokenizer=tokenizer,
            max_length=max_length,
            max_rationale_length=max_rationale_length,
            text_column="text",
            label_column="label",
            rationale_column="rationale",
            include_rationales=include_rationales,
        )

    def _preprocess_task_data(
        self,
        data: Union[str, Path, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Preprocess task-specific data into unified format."""
        # Load raw data
        if isinstance(data, (str, Path)):
            raw_data = self._load_data(data)
        else:
            raw_data = data

        processed = []
        task_format = self.TASK_FORMATS[self.task_name]

        for item in raw_data:
            # Format text based on task
            if self.task_name in ["cb", "rte"]:
                text = f"Premise: {item[task_format['text']]}\nHypothesis: {item[task_format['text2']]}"
            elif self.task_name == "copa":
                question = item[task_format['question']]
                premise = item[task_format['text']]
                choice1 = item[task_format['choice1']]
                choice2 = item[task_format['choice2']]
                text = f"{premise}\nQuestion: {question}\nChoice 1: {choice1}\nChoice 2: {choice2}"
            elif self.task_name == "boolq":
                text = f"Passage: {item[task_format['text']]}\nQuestion: {item[task_format['question']]}"
            elif self.task_name == "wic":
                text = f"Word: {item[task_format['word']]}\nSentence 1: {item[task_format['text']]}\nSentence 2: {item[task_format['text2']]}"
            else:
                text = str(item[task_format['text']])

            processed_item = {
                "text": text,
                "label": item[task_format['label']],
            }

            # Add rationale if available
            if "rationale" in item:
                processed_item["rationale"] = item["rationale"]

            processed.append(processed_item)

        return processed


def create_dhrd_dataloader(
    dataset: DHRDDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for DHRD training.

    Args:
        dataset: DHRDDataset instance
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
