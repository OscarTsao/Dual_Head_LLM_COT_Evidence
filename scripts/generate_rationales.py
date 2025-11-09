#!/usr/bin/env python3
"""
Generate rationales for training data using a teacher model.

This script uses a capable LM (e.g., GPT-4, Gemini, etc.) to generate
Chain-of-Thought rationales for classification tasks.

Usage:
    python scripts/generate_rationales.py --input data/train.json --output data/train_with_rationales.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import time

# For local models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# For API-based models (OpenAI, Anthropic, etc.)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RationaleGenerator:
    """
    Base class for generating rationales using teacher models.
    """

    def generate(self, text: str, label: Optional[int] = None) -> str:
        """
        Generate a rationale for the given text.

        Args:
            text: Input text
            label: (Optional) Ground truth label

        Returns:
            Generated rationale
        """
        raise NotImplementedError


class LocalModelGenerator(RationaleGenerator):
    """
    Generate rationales using a local transformers model.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Args:
            model_name: Name of the model to use
            device: Device to run model on
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_prompt(self, text: str, label: Optional[int] = None) -> str:
        """Create a Chain-of-Thought prompt."""
        prompt = f"""Given the following text, provide a step-by-step reasoning to classify it.

Text: {text}

Let's think step by step to determine the correct classification:"""

        return prompt

    def generate(self, text: str, label: Optional[int] = None) -> str:
        """Generate rationale using the local model."""
        prompt = self.create_prompt(text, label)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        rationale = generated_text[len(prompt):].strip()

        return rationale


class OpenAIGenerator(RationaleGenerator):
    """
    Generate rationales using OpenAI API (GPT-4, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available. Install with: pip install openai")

        if api_key:
            openai.api_key = api_key

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create_prompt(self, text: str, label: Optional[int] = None) -> str:
        """Create a Chain-of-Thought prompt."""
        return f"""Given the following text, provide a step-by-step reasoning to classify it.

Text: {text}

Let's think step by step to determine the correct classification:"""

    def generate(self, text: str, label: Optional[int] = None) -> str:
        """Generate rationale using OpenAI API."""
        prompt = self.create_prompt(text, label)

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides clear, step-by-step reasoning for text classification tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            rationale = response.choices[0].message.content.strip()
            return rationale

        except Exception as e:
            logger.error(f"Error generating rationale: {e}")
            return ""


class ZeroShotCoTGenerator(RationaleGenerator):
    """
    Generate rationales using Zero-shot Chain-of-Thought prompting.
    Can be used with any API or local model.
    """

    def __init__(self, base_generator: RationaleGenerator):
        """
        Args:
            base_generator: Base generator to use (LocalModel or OpenAI)
        """
        self.base_generator = base_generator

    def create_prompt(self, text: str, label: Optional[int] = None) -> str:
        """Create a Zero-shot CoT prompt with 'Let's think step by step'."""
        return f"""{text}

Let's think step by step:"""

    def generate(self, text: str, label: Optional[int] = None) -> str:
        """Generate rationale using Zero-shot CoT."""
        # Override the base generator's prompt creation
        original_create_prompt = self.base_generator.create_prompt
        self.base_generator.create_prompt = self.create_prompt

        rationale = self.base_generator.generate(text, label)

        # Restore original prompt creation
        self.base_generator.create_prompt = original_create_prompt

        return rationale


def load_data(input_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    input_path = Path(input_path)

    if input_path.suffix == ".json":
        with open(input_path, 'r') as f:
            return json.load(f)
    elif input_path.suffix == ".jsonl":
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def save_data(data: List[Dict[str, Any]], output_path: str):
    """Save data to JSON or JSONL file."""
    output_path = Path(output_path)

    if output_path.suffix == ".json":
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif output_path.suffix == ".jsonl":
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate rationales for training data using a teacher model"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data file (JSON/JSONL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output data file (JSON/JSONL)"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="local",
        choices=["local", "openai"],
        help="Type of generator to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (for local) or model ID (for OpenAI)"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column in data"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of label column in data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--batch_delay",
        type=float,
        default=0.0,
        help="Delay between batches in seconds (for API rate limiting)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("Starting rationale generation")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Load data
    logger.info("Loading data...")
    data = load_data(args.input)
    logger.info(f"Loaded {len(data)} samples")

    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"Processing {len(data)} samples")

    # Create generator
    if args.generator == "local":
        if not args.model:
            args.model = "meta-llama/Llama-2-7b-chat-hf"
        generator = LocalModelGenerator(model_name=args.model)
    elif args.generator == "openai":
        if not args.model:
            args.model = "gpt-4"
        generator = OpenAIGenerator(model=args.model)
    else:
        raise ValueError(f"Unknown generator: {args.generator}")

    # Use Zero-shot CoT
    generator = ZeroShotCoTGenerator(generator)

    # Generate rationales
    logger.info("Generating rationales...")
    for i, item in enumerate(tqdm(data)):
        if "rationale" in item and item["rationale"]:
            # Skip if rationale already exists
            continue

        text = item[args.text_column]
        label = item.get(args.label_column)

        # Generate rationale
        try:
            rationale = generator.generate(text, label)
            item["rationale"] = rationale

            # Rate limiting delay
            if args.batch_delay > 0:
                time.sleep(args.batch_delay)

        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            item["rationale"] = ""

    # Save data
    logger.info(f"Saving data to {args.output}")
    save_data(data, args.output)
    logger.info("Done!")


if __name__ == "__main__":
    main()
