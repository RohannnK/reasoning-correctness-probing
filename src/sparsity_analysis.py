"""Train L1-regularized probes and analyze sparsity of correctness representations."""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import load_hidden_states


def sparsity_analysis(
    hidden_states_dir: str,
    model_short_name: str,
    output_path: str,
    C_values: list[float] | None = None,
    top_k: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train L1-regularized probes and report sparsity metrics per layer.

    Args:
        hidden_states_dir: Directory containing npz files.
        model_short_name: Model short name for file matching.
        output_path: Path to save results CSV.
        C_values: Regularization values to try.
        top_k: Number of top important dimensions to report.
        test_size: Fraction of data for test set.
        random_state: Random seed.

    Returns:
        DataFrame with sparsity results per layer.
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    print(f"Loading hidden states for {model_short_name}...")
    layer_data = load_hidden_states(hidden_states_dir, model_short_name)

    results = []

    for layer_idx, (X, y) in sorted(layer_data.items()):
        if len(np.unique(y)) < 2:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for C in C_values:
            probe = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=C,
                max_iter=2000,
                random_state=random_state,
            )
            probe.fit(X_train_scaled, y_train)

            coefs = probe.coef_[0]
            n_nonzero = int(np.count_nonzero(coefs))
            n_total = len(coefs)
            sparsity_ratio = 1.0 - (n_nonzero / n_total)

            # Top-k most important dimensions by absolute coefficient
            top_indices = np.argsort(np.abs(coefs))[-top_k:][::-1]
            top_dims = top_indices.tolist()

            accuracy = probe.score(X_test_scaled, y_test)

            results.append({
                "model": model_short_name,
                "layer": layer_idx,
                "C": C,
                "n_nonzero": n_nonzero,
                "n_total": n_total,
                "sparsity_ratio": sparsity_ratio,
                "test_accuracy": accuracy,
                "top_k_dims": str(top_dims),
            })

        print(f"  Layer {layer_idx}: best nonzero = "
              f"{min(r['n_nonzero'] for r in results[-len(C_values):])} / {n_total}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved sparsity results to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="L1 sparsity analysis of probes")
    parser.add_argument("--hidden_states_dir", type=str, required=True)
    parser.add_argument("--model_short_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    sparsity_analysis(
        hidden_states_dir=args.hidden_states_dir,
        model_short_name=args.model_short_name,
        output_path=args.output_path,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
