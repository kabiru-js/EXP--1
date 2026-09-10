"""Dataset loaders."""

from src.datasets.registry import get_loader, load_dataset_by_name, available_datasets

__all__ = ["get_loader", "load_dataset_by_name", "available_datasets"]
