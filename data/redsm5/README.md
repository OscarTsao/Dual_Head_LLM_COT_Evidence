# REDSM5 Evidence Binding Dataset

This directory contains data for the **Evidence Binding** task using DHRD (Dual-Head Reasoning Distillation).

## Overview

Evidence binding involves identifying and extracting key evidence sentences that support or refute claims in medical/scientific texts. This task uses the DHRD architecture to:

1. **Input**: Medical claims and contextual information
2. **Task**: Identify relevant evidence sentences
3. **Output**: Binary classification (has evidence: 1, no evidence: 0) + extracted evidence text

## Data Format

### annotations.csv

The main data file with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `id` | Unique identifier for each sample | Yes |
| `status` | Status flag (1 = valid, 0 = invalid/filtered) | Yes |
| `claim` | The main claim to be evaluated | Recommended |
| `context` | Contextual information or background | Recommended |
| `sentence` | **Evidence ground truth** - the key evidence sentence | **Yes** |
| `rationale` | (Optional) Reasoning explanation for the evidence | No |

**Important Notes:**
- Only rows with `status=1` are used for training
- The `sentence` column contains the **evidence ground truth**
- Empty `sentence` values indicate no evidence exists (label = 0)
- Non-empty `sentence` values indicate evidence exists (label = 1)

### Example Row

```csv
id,status,claim,context,sentence,rationale
1,1,"Patient has diabetes","Patient presents with elevated blood glucose levels of 180 mg/dL, increased thirst.","Patient presents with elevated blood glucose levels of 180 mg/dL","Blood glucose level above 126 mg/dL is diagnostic of diabetes."
```

## Quick Start

### 1. Prepare Your Data

Place your `annotations.csv` file in this directory:
```bash
data/redsm5/annotations.csv
```

### 2. Preprocess the Data

Split the data into train/val/test sets:

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

This will create:
- `train_annotations.csv` (70% of data)
- `val_annotations.csv` (15% of data)
- `test_annotations.csv` (15% of data)

### 3. Train DHRD Model

Train the evidence binding model:

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

### 4. Run Inference

Use the trained model for predictions:

```bash
python scripts/inference_dhrd.py \
  --model_path outputs/evidence_binding \
  --input "Patient presents with fever and elevated WBC count of 15,000." \
  --batch_size 32
```

## Data Customization

### Custom Column Names

If your CSV has different column names, specify them:

```bash
python scripts/train_evidence_binding.py \
  --data_dir data/redsm5 \
  --evidence_column "your_evidence_column" \
  --model gpt2
```

### Custom Text Columns

The dataset loader automatically uses all columns (except metadata) for input text. To specify custom columns, modify the `EvidenceBindingDataset` initialization in your training script:

```python
from Project.SubProject.data.evidence_binding_dataset import EvidenceBindingDataset

dataset = EvidenceBindingDataset(
    csv_path="data/redsm5/train_annotations.csv",
    tokenizer=tokenizer,
    text_columns=["claim", "context"],  # Specify custom columns
    evidence_column="sentence",
    status_filter=None,  # Already filtered
)
```

## Expected Performance

Based on the DHRD paper, you can expect:

- **Accuracy Improvement**: 0.65-5.47% over pooled baselines
- **Inference Speed**: 96-142× faster than Chain-of-Thought decoding
- **Best Results**: On tasks requiring reasoning about evidence (like evidence binding)

## Dataset Statistics

After preprocessing, check statistics:

```bash
python scripts/prepare_redsm5.py \
  --input data/redsm5/annotations.csv \
  --output data/redsm5 \
  --show-stats
```

This shows:
- Total samples
- Status distribution
- Evidence present/missing counts
- Text length statistics
- Sample rows

## Troubleshooting

### "Training file not found" error

**Solution**: Run the preprocessing script first:
```bash
python scripts/prepare_redsm5.py --input data/redsm5/annotations.csv --output data/redsm5
```

### "No data remaining after filtering" error

**Cause**: No rows with `status=1` in your CSV

**Solutions**:
1. Check your CSV has rows with `status=1`
2. Change `--status-filter` value
3. Set `status_filter=None` to use all rows

### Missing columns error

**Solution**: Ensure your CSV has at minimum:
- `status` column
- `sentence` column (evidence ground truth)

## Example Data

An example dataset (`example_annotations.csv`) is provided in this directory showing the expected format with:
- Medical claims
- Contextual information
- Evidence sentences
- Status flags
- Optional rationales

## Advanced Usage

### Multi-class Classification

For non-binary classification (e.g., evidence strength levels):

```python
dataset = EvidenceBindingDataset(
    csv_path="data/redsm5/train_annotations.csv",
    tokenizer=tokenizer,
    binary_classification=False,
    label_column="evidence_strength",  # Custom label column
)
```

### Custom Rationale Generation

The dataset auto-generates rationales from evidence sentences. For custom rationales:

1. Add a `rationale` column to your CSV
2. The dataset will use it automatically

Or generate rationales with a teacher model:

```bash
python scripts/generate_rationales.py \
  --input data/redsm5/train_annotations.csv \
  --output data/redsm5/train_with_rationales.csv \
  --generator openai \
  --model gpt-4
```

## Citation

If you use this evidence binding implementation, please cite:

```bibtex
@article{xu2025dhrd,
  title={Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning},
  author={Xu, Jillian and others},
  journal={arXiv preprint arXiv:2509.21487},
  year={2025}
}
```

## Support

For issues or questions:
1. Check the main [README](../../README.md)
2. Review this documentation
3. Open an issue on GitHub

---

**Note**: This is an implementation for the evidence binding task using DHRD. Ensure your data complies with relevant privacy and usage regulations.
