# Evidence Binding with DHRD (REDSM5 Dataset)

[![arXiv](https://img.shields.io/badge/arXiv-2509.21487-b31b1b.svg)](https://arxiv.org/abs/2509.21487)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

Implementation of **Evidence Binding** using **Dual-Head Reasoning Distillation (DHRD)** for the REDSM5 dataset.

## Overview

This repository implements evidence binding for medical/scientific text using the DHRD architecture from the paper *"Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning"* (arXiv:2509.21487).

### What is Evidence Binding?

**Evidence binding** identifies and extracts key evidence sentences that support or refute claims in medical/scientific texts:
- **Input**: Claims and contextual information from medical cases
- **Task**: Identify whether relevant evidence exists and extract it
- **Output**: Binary classification (has_evidence: 0/1) + evidence sentence extraction

### DHRD Architecture

DHRD uses a dual-head approach for improved accuracy:

1. **Classification Head** (training + inference): Pooled classifier for evidence detection
2. **Reasoning Head** (training only): LM head that learns from evidence rationales

**Key Benefits**:
- **Improved Accuracy**: 0.65-5.47% gains over baselines (especially on reasoning tasks)
- **Fast Inference**: 96-142× faster than Chain-of-Thought decoding
- **Train-Time-Only Reasoning**: Reasoning head disabled at inference for speed

```
Training:
Medical Text → Transformer → Classification Head → Has Evidence? (0/1)
                           ↓
                           Reasoning Head → Learn from Evidence Rationale

Inference (Fast):
Medical Text → Transformer → Classification Head → Has Evidence? (0/1)
                           (Reasoning Head Disabled)
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.2+
- Transformers 4.40+
- CUDA (recommended for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Dual_Head_LLM_COT_Evidence.git
cd Dual_Head_LLM_COT_Evidence

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Optional: Install development dependencies
pip install -e '.[dev]'

# Optional: Install API dependencies for rationale generation
pip install -e '.[api]'
```

## Quick Start (REDSM5 Evidence Binding)

### 1. Prepare Your Data

Place your `annotations.csv` file in `data/redsm5/`:

```csv
id,status,claim,context,sentence
1,1,"Patient has diabetes","Patient presents with elevated blood glucose levels of 180 mg/dL","Patient presents with elevated blood glucose levels of 180 mg/dL"
2,0,"Patient has flu","General malaise",""
```

**Required columns**:
- `status`: Filter flag (1 = use this row, 0 = skip)
- `sentence`: **Evidence ground truth** (the key evidence text)
- Other columns: Automatically used as input context

See `data/redsm5/README.md` for detailed format specifications and `data/redsm5/example_annotations.csv` for examples.

### 2. Preprocess the Data

Split your annotations into train/val/test sets:

```bash
python scripts/prepare_redsm5.py \
  --input data/redsm5/annotations.csv \
  --output data/redsm5 \
  --status-filter 1 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --show-stats
```

This creates:
- `train_annotations.csv` (70% of data)
- `val_annotations.csv` (15% of data)
- `test_annotations.csv` (15% of data)

### 3. Train Evidence Binding Model

```bash
python scripts/train_evidence_binding.py \
  --data_dir data/redsm5 \
  --model gpt2 \
  --evidence_column sentence \
  --num_epochs 5 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --output_dir outputs/evidence_binding \
  --use_mlflow
```

**Model options**:
- `--model`: gpt2, facebook/opt-125m, facebook/opt-350m, etc.
- `--reasoning_weight`: 0.5 (default) - balance between classification and reasoning
- `--max_length`: 512 (default) - max input sequence length

### 4. Run Inference

```bash
# Inference on test set
python scripts/inference_dhrd.py \
  --model_path outputs/evidence_binding \
  --input_file data/redsm5/test_annotations.csv \
  --output_file results/evidence_predictions.json \
  --batch_size 32

# Single case inference
python scripts/inference_dhrd.py \
  --model_path outputs/evidence_binding \
  --input "Patient presents with fever and WBC count of 15,000"
```

## Architecture Details

### Model Components

#### 1. Classification Head

```python
ClassificationHead(
    hidden_size: int,      # Transformer hidden size
    num_labels: int,       # Number of classes
    dropout_prob: float,   # Dropout probability
    pooling_method: str    # 'last' or 'mean'
)
```

Features:
- Last-token pooling (default) or mean pooling
- Dropout regularization
- Linear projection to label space

#### 2. Reasoning Head

```python
ReasoningHead(
    hidden_size: int,      # Transformer hidden size
    vocab_size: int        # Vocabulary size
)
```

Features:
- Standard LM head for token prediction
- Used only during training
- Supervised by teacher rationales

#### 3. Combined DHRD Model

```python
DHRDModel(
    backbone_model_name: str,     # e.g., 'gpt2', 'facebook/opt-125m'
    num_labels: int,              # Number of classes
    reasoning_weight: float,      # λ for loss combination (0.0-1.0)
    pooling_method: str,          # 'last' or 'mean'
    classification_dropout: float # Dropout probability
)
```

### Training Process

1. **Forward pass through classification head**:
   - Input → Transformer → Pool hidden states → Classification logits
   - Compute classification loss (cross-entropy)

2. **Forward pass through reasoning head** (if rationales available):
   - Input + Rationale → Transformer → LM logits
   - Compute reasoning loss (token-level cross-entropy)

3. **Combine losses**:
   - Total loss = (1-λ) × Classification loss + λ × Reasoning loss

4. **Gradient update and optimization**

### Inference Process

At inference time:
- Reasoning head is **disabled** for maximum speed
- Only classification head is used
- Throughput matches standard pooled classifiers
- 96-142× faster than CoT decoding

## Configuration

Configuration files are in YAML format. See `configs/dhrd_config.yaml` for a full example:

```yaml
model:
  backbone: "gpt2"
  num_labels: 3
  reasoning_weight: 0.5
  pooling_method: "last"

training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 2.0e-5
  warmup_ratio: 0.1

data:
  train_path: "data/train_with_rationales.json"
  val_path: "data/val.json"
  max_length: 512
  max_rationale_length: 1024
```

## Supported Backbones

DHRD works with any decoder-only transformer model from Hugging Face:

- **GPT-2**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **OPT**: `facebook/opt-125m`, `facebook/opt-350m`, `facebook/opt-1.3b`
- **GPT-Neo**: `EleutherAI/gpt-neo-125M`, `EleutherAI/gpt-neo-1.3B`
- **GPT-J**: `EleutherAI/gpt-j-6B`
- **LLaMA**: `meta-llama/Llama-2-7b-hf`, etc.

## Project Structure (REDSM5 Evidence Binding)

```
Dual_Head_LLM_COT_Evidence/
├── src/
│   └── Project/
│       └── SubProject/
│           ├── models/
│           │   └── dhrd_model.py                # DHRD architecture
│           ├── data/
│           │   └── evidence_binding_dataset.py  # REDSM5 dataset loader
│           ├── engine/
│           │   └── dhrd_trainer.py              # Training engine
│           └── utils/
│               ├── log.py                       # Logging utilities
│               ├── seed.py                      # Seed setting
│               └── mlflow_utils.py              # MLflow helpers
├── scripts/
│   ├── prepare_redsm5.py                        # Data preprocessing
│   ├── train_evidence_binding.py                # Training script
│   ├── inference_dhrd.py                        # Inference script
│   └── generate_rationales.py                   # Rationale generation (optional)
├── configs/
│   └── evidence_binding_config.yaml             # Evidence binding config
├── data/
│   └── redsm5/                                  # REDSM5 dataset
│       ├── README.md                            # Data format guide
│       ├── example_annotations.csv              # Example data
│       └── annotations.csv                      # Your data (place here)
├── outputs/                                     # Model outputs
├── mlruns/                                      # MLflow tracking
├── pyproject.toml                               # Project metadata
├── LICENSE                                      # MIT License
└── README.md                                    # This file
```

## Performance

Based on the paper (arXiv:2509.21487):

### SuperGLUE Results

| Task | Baseline Accuracy | DHRD Accuracy | Relative Gain |
|------|------------------|---------------|---------------|
| CB   | 85.2%           | 89.9%         | +5.47%        |
| RTE  | 77.6%           | 81.2%         | +4.64%        |
| COPA | 82.4%           | 86.1%         | +4.49%        |
| BoolQ| 81.8%           | 83.2%         | +1.71%        |
| WiC  | 69.5%           | 70.0%         | +0.72%        |

### Inference Speed

- **Classification**: Same as pooled baseline
- **vs. CoT Decoding**: 96-142× faster (QPS improvement)
- **Reasoning Head**: Disabled at test time (no overhead)

## Citation

If you use this code or find it helpful, please cite the original paper:

```bibtex
@article{xu2025dhrd,
  title={Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning},
  author={Xu, Jillian and others},
  journal={arXiv preprint arXiv:2509.21487},
  year={2025}
}
```

## Paper Link

**arXiv**: [https://arxiv.org/abs/2509.21487](https://arxiv.org/abs/2509.21487)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black src/ scripts/ tests/

# Check linting
ruff check src/ scripts/ tests/
```

### Type Checking

```bash
mypy src/
```

## MLflow Tracking

Track experiments with MLflow:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Open browser to http://localhost:5000
```

MLflow automatically logs:
- Training/validation losses
- Classification and reasoning losses
- Learning rates
- Model checkpoints
- Hyperparameters
- Evaluation metrics

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size`
- Reduce `max_length` or `max_rationale_length`
- Use gradient accumulation
- Use smaller backbone model

### Slow Training

- Increase `batch_size` if memory allows
- Reduce `eval_interval` and `log_interval`
- Use smaller validation set
- Disable MLflow logging

### Poor Performance

- Increase `num_epochs`
- Adjust `learning_rate` (try 1e-5 to 5e-5)
- Tune `reasoning_weight` (try 0.3 to 0.7)
- Use better teacher model for rationales
- Check data quality and rationales

## Advanced Usage

### Custom Loss Weights

```python
from Project.SubProject.models.dhrd_model import DHRDModel

model = DHRDModel(
    backbone_model_name="gpt2",
    num_labels=3,
    reasoning_weight=0.7,  # Higher weight on reasoning
)
```

### Custom Pooling

```python
model = DHRDModel(
    backbone_model_name="gpt2",
    num_labels=3,
    pooling_method="mean",  # Use mean pooling instead of last-token
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper authors: Jillian Xu et al.
- Hugging Face Transformers library
- PyTorch team

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is an implementation of the DHRD paper for research and educational purposes. For the official implementation (if available), please refer to the paper's authors.
