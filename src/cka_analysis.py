"""Compute linear CKA between base and distilled model representations."""

import argparse
import os

import numpy as np
import pandas as pd

from src.utils import load_hidden_states


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two representation matrices.

    Args:
        X: Representation matrix of shape (n_samples, d1).
        Y: Representation matrix of shape (n_samples, d2).

    Returns:
        Linear CKA similarity value in [0, 1].
    """
    # Center the representations
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0

    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def compute_cka(
    hidden_states_dir: str,
    base_short_name: str,
    distilled_short_name: str,
    output_path: str,
) -> pd.DataFrame:
    """Compute layer-wise linear CKA between base and distilled models.

    Both models must have hidden states extracted for the same problems.

    Args:
        hidden_states_dir: Directory containing npz files for both models.
        base_short_name: Short name of the base model.
        distilled_short_name: Short name of the distilled model.
        output_path: Path to save the CKA results CSV.

    Returns:
        DataFrame with layer-wise CKA values.
    """
    print(f"Loading hidden states for {base_short_name}...")
    base_data = load_hidden_states(hidden_states_dir, base_short_name)

    print(f"Loading hidden states for {distilled_short_name}...")
    distilled_data = load_hidden_states(hidden_states_dir, distilled_short_name)

    # Use layers present in both models
    common_layers = sorted(set(base_data.keys()) & set(distilled_data.keys()))
    print(f"Computing CKA for {len(common_layers)} common layers")

    results = []
    for layer_idx in common_layers:
        X_base, _ = base_data[layer_idx]
        X_dist, _ = distilled_data[layer_idx]

        # Align samples: use the minimum number of samples
        n = min(X_base.shape[0], X_dist.shape[0])
        cka_val = linear_cka(X_base[:n], X_dist[:n])

        results.append({
            "layer": layer_idx,
            "cka": cka_val,
        })
        print(f"  Layer {layer_idx}: CKA = {cka_val:.4f}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved CKA results to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute linear CKA between models")
    parser.add_argument("--hidden_states_dir", type=str, required=True)
    parser.add_argument("--base_short_name", type=str, required=True)
    parser.add_argument("--distilled_short_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    compute_cka(
        hidden_states_dir=args.hidden_states_dir,
        base_short_name=args.base_short_name,
        distilled_short_name=args.distilled_short_name,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
