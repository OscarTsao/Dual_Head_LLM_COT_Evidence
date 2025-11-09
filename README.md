# AI/ML Experiment Template

Minimal template for ML experiments using PyTorch, Transformers, MLflow, and Optuna.

## Quickstart

- Python 3.10+ recommended.
- Create and activate a virtual environment, then install:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Layout

- `src/Project/SubProject/models/model.py` – example model wrapper for Transformers.
- `src/Project/SubProject/utils/` – utility helpers (`get_logger`, `set_seed`, MLflow helpers).
- `mlruns/` – local MLflow runs (if using file-based tracking).
- `outputs/` – suggested place for artifacts.

## MLflow

Configure MLflow for local tracking and run logging:

```python
from Project.SubProject.utils import configure_mlflow, enable_autologging, mlflow_run

configure_mlflow(tracking_uri="file:./mlruns", experiment="demo")
enable_autologging()

with mlflow_run("hello", tags={"stage": "dev"}, params={"lr": 1e-4}):
    # your training loop here
    pass
```

## Development

- Run linters/formatters:
```
ruff check src tests
black src tests
```
- Run tests (add your own under `tests/`):
```
pytest
```

