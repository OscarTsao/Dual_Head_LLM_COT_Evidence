from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional


def configure_mlflow(
    tracking_uri: Optional[str] = None,
    experiment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Configure MLflow tracking URI and experiment.

    - `tracking_uri`: e.g. "file:./mlruns" or a remote server URI
    - `experiment`: experiment name (created if missing)
    - `tags`: default tags set for the active run (if any)
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)
    if tags:
        try:
            mlflow.set_tags(tags)
        except Exception:
            # set_tags requires an active run; ignore if none is active
            pass


def enable_autologging(enable: bool = True) -> None:
    """Enable or disable MLflow autologging with sensible defaults.

    Attempts framework-specific autologging when available and falls back
    to generic `mlflow.autolog`.
    """
    import mlflow

    if not enable:
        mlflow.autolog(disable=True)
        return

    # Prefer generic autolog (works for many frameworks in MLflow>=2)
    try:
        mlflow.autolog()
        return
    except Exception:
        pass

    # Fall back to common framework-specific autologging if present
    for mod_name in ("mlflow.pytorch", "mlflow.sklearn", "mlflow.xgboost", "mlflow.lightgbm"):
        try:
            mod = __import__(mod_name, fromlist=["autolog"])  # type: ignore
            getattr(mod, "autolog")()
        except Exception:
            continue


@contextlib.contextmanager
def mlflow_run(
    name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Context manager that starts and ends an MLflow run.

    Usage:
        with mlflow_run("demo", tags={"stage": "dev"}):
            ... your training loop ...
    """
    import mlflow

    with mlflow.start_run(run_name=name, nested=nested) as run:
        if tags:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass
        if params:
            try:
                mlflow.log_params(params)
            except Exception:
                pass
        yield run

