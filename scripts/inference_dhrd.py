#!/usr/bin/env python3
"""
Inference script for Dual-Head Reasoning Distillation (DHRD) models.

This script demonstrates fast inference using trained DHRD models
with the reasoning head disabled.

Usage:
    python scripts/inference_dhrd.py --model_path outputs/dhrd --input "Your text here"
    python scripts/inference_dhrd.py --model_path outputs/dhrd --input_file data/test.json
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.SubProject.models.dhrd_model import DHRDModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DHRDInference:
    """
    Inference wrapper for DHRD models.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """
        Args:
            model_path: Path to saved model directory
            device: Device to run inference on
            batch_size: Batch size for inference
        """
        self.device = device
        self.batch_size = batch_size

        # Load model configuration
        config_path = Path(model_path) / "training_args.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            logger.warning(f"Config file not found at {config_path}")
            self.config = {}

        # Load tokenizer
        tokenizer_path = Path(model_path) / "tokenizer"
        if tokenizer_path.exists():
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            # Fallback to model name from config
            model_name = self.config.get("model", "gpt2")
            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create model
        logger.info(f"Creating DHRD model")
        self.model = DHRDModel(
            backbone_model_name=self.config.get("model", "gpt2"),
            num_labels=self.config.get("num_labels", 2),
            reasoning_weight=self.config.get("reasoning_weight", 0.5),
            pooling_method=self.config.get("pooling_method", "last"),
            use_pretrained=False,  # We'll load weights manually
        )

        # Load model weights
        model_weights_path = Path(model_path) / "final_model.pt"
        if not model_weights_path.exists():
            model_weights_path = Path(model_path) / "best_model.pt"

        if model_weights_path.exists():
            logger.info(f"Loading model weights from {model_weights_path}")
            state_dict = torch.load(model_weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        # Move model to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()
        self.model.disable_reasoning_head()  # Disable reasoning for fast inference

        logger.info(f"Model loaded successfully on {device}")

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict on a single text input.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.get("max_length", 512),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Get prediction
        probs = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, predicted_label].item()

        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probs[0].cpu().numpy().tolist(),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict on a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of prediction results
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.config.get("max_length", 512),
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Predict
            with torch.no_grad():
                logits = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(logits, dim=-1)

            # Collect results
            for j, text in enumerate(batch_texts):
                results.append({
                    "text": text,
                    "predicted_label": predicted_labels[j].item(),
                    "confidence": probs[j, predicted_labels[j]].item(),
                    "probabilities": probs[j].cpu().numpy().tolist(),
                })

        return results

    def predict_file(self, input_file: str, text_column: str = "text") -> List[Dict[str, Any]]:
        """
        Predict on data from a file.

        Args:
            input_file: Path to input file (JSON/JSONL)
            text_column: Name of text column

        Returns:
            List of prediction results
        """
        # Load data
        input_path = Path(input_file)
        if input_path.suffix == ".json":
            with open(input_path, 'r') as f:
                data = json.load(f)
        elif input_path.suffix == ".jsonl":
            data = []
            with open(input_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        # Extract texts
        texts = [item[text_column] for item in data]

        # Predict
        logger.info(f"Predicting on {len(texts)} samples")
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self.predict_batch(batch_texts)
            results.extend(batch_results)

        # Add original data
        for i, result in enumerate(results):
            result.update(data[i])

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference with DHRD model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model directory"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Single text input for inference"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input file (JSON/JSONL) for batch inference"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save prediction results"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column in input file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    # Create inference wrapper
    logger.info("Loading model...")
    inference = DHRDInference(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Single text inference
    if args.input:
        logger.info(f"Running inference on single text")
        result = inference.predict_single(args.input)

        print("\n" + "="*50)
        print("Inference Result:")
        print("="*50)
        print(f"Text: {result['text']}")
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
        print("="*50 + "\n")

    # Batch inference from file
    elif args.input_file:
        logger.info(f"Running inference on file: {args.input_file}")
        results = inference.predict_file(args.input_file, args.text_column)

        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
            if output_path.suffix == ".json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            elif output_path.suffix == ".jsonl":
                with open(output_path, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            logger.info(f"Results saved to {args.output_file}")
        else:
            # Print first few results
            print("\n" + "="*50)
            print(f"Inference Results (showing first 5 of {len(results)}):")
            print("="*50)
            for i, result in enumerate(results[:5]):
                print(f"\nSample {i+1}:")
                print(f"  Text: {result['text'][:100]}...")
                print(f"  Predicted Label: {result['predicted_label']}")
                print(f"  Confidence: {result['confidence']:.4f}")
            print("="*50 + "\n")

    else:
        logger.error("Either --input or --input_file must be provided")
        return

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
