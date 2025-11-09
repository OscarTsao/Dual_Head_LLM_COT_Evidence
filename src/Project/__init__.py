"""AI/ML Experiment Template - Source Package."""

__version__ = "0.1.0"

# Keep surface minimal in template state to avoid import errors.
# Expose existing subpackages explicitly as they are added.
from . import SubProject  # noqa: F401

__all__ = [
    "SubProject",
    "__version__",
]
