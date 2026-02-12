"""Shared utilities for loading hidden states, computing ECE, and formatting results."""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


def load_hidden_states(
    hidden_states_dir: str,
    model_short_name: Optional[str] = None,
    problem_ids: Optional[list[str]] = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load hidden states from npz files and aggregate by layer.

    Args:
        hidden_states_dir: Directory containing npz files (e.g. data/hidden_states/distilled/).
        model_short_name: Deprecated. Kept for backwards compatibility but ignored.
            The directory should already point to the model-specific subdirectory.
        problem_ids: If provided, only load these problem IDs.

    Returns:
        Dictionary mapping layer index to (features, labels) arrays.
    """
    layer_features: dict[int, list[np.ndarray]] = {}
    layer_labels: dict[int, list[np.ndarray]] = {}

    directory = Path(hidden_states_dir)
    pattern = "problem_*.npz"

    for npz_path in sorted(directory.glob(pattern)):
        pid = npz_path.stem.split("problem_")[1]
        if problem_ids is not None and pid not in problem_ids:
            continue

        data = np.load(npz_path)
        labels = data["labels"]

        for key in data.files:
            if key == "labels":
                continue
            layer_idx = int(key.replace("layer_", ""))
            if layer_idx not in layer_features:
                layer_features[layer_idx] = []
                layer_labels[layer_idx] = []
            layer_features[layer_idx].append(data[key])
            layer_labels[layer_idx].append(labels)

    result = {}
    for layer_idx in sorted(layer_features.keys()):
        X = np.concatenate(layer_features[layer_idx], axis=0)
        y = np.concatenate(layer_labels[layer_idx], axis=0)
        result[layer_idx] = (X, y)

    return result


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins for calibration.

    Returns:
        ECE value.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    """Save a list of dicts as a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def format_metrics(metrics: dict) -> str:
    """Format a metrics dictionary into a readable string."""
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
