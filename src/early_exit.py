"""Simulate threshold-based early exit using trained probes."""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.utils import load_hidden_states


def simulate_early_exit(
    hidden_states_dir: str,
    probes_dir: str,
    model_short_name: str,
    output_path: str,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Simulate early exit by checking probe confidence at each layer.

    For each threshold, iterates through layers from earliest to latest. If the
    probe's confidence (max predicted probability) exceeds the threshold, the
    model "exits" at that layer. Computes accuracy and percentage of tokens
    (layers) saved compared to running through all layers.

    Args:
        hidden_states_dir: Directory containing npz files.
        probes_dir: Directory containing trained probe checkpoints.
        model_short_name: Model short name.
        output_path: Path to save Pareto frontier CSV.
        thresholds: Confidence thresholds to evaluate.

    Returns:
        DataFrame with early exit results.
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    print(f"Loading hidden states for {model_short_name}...")
    layer_data = load_hidden_states(hidden_states_dir, model_short_name)
    layers = sorted(layer_data.keys())
    num_layers = len(layers)

    if num_layers == 0:
        print("No layers found.")
        return pd.DataFrame()

    # Load all LR probes
    probes: dict[int, dict] = {}
    for layer_idx in layers:
        probe_path = os.path.join(probes_dir, f"{model_short_name}_lr_layer_{layer_idx}.joblib")
        if os.path.exists(probe_path):
            probes[layer_idx] = joblib.load(probe_path)

    if not probes:
        print("No trained probes found.")
        return pd.DataFrame()

    # Get ground truth labels (same across layers)
    _, y_true = layer_data[layers[0]]
    n_samples = len(y_true)

    results = []

    for threshold in thresholds:
        exit_layers = np.full(n_samples, layers[-1])  # default: last layer
        predictions = np.zeros(n_samples, dtype=int)

        # Track which samples have already exited
        exited = np.zeros(n_samples, dtype=bool)

        for layer_idx in layers:
            if layer_idx not in probes:
                continue

            probe_data = probes[layer_idx]
            probe = probe_data["probe"]
            scaler = probe_data["scaler"]

            X, _ = layer_data[layer_idx]
            X_scaled = scaler.transform(X)
            probs = probe.predict_proba(X_scaled)

            # Confidence = max probability across classes
            confidence = probs.max(axis=1)

            # Samples that should exit at this layer
            should_exit = (~exited) & (confidence >= threshold)
            if should_exit.any():
                exit_layers[should_exit] = layer_idx
                predictions[should_exit] = probe.predict(X_scaled[should_exit])
                exited[should_exit] = True

        # Handle samples that never exceeded the threshold: use last layer
        if not exited.all():
            last_layer = layers[-1]
            if last_layer in probes:
                probe_data = probes[last_layer]
                X, _ = layer_data[last_layer]
                X_scaled = probe_data["scaler"].transform(X)
                remaining = ~exited
                predictions[remaining] = probe_data["probe"].predict(X_scaled[remaining])

        accuracy = accuracy_score(y_true, predictions)
        avg_exit_layer = exit_layers.mean()
        layers_saved_pct = (1.0 - avg_exit_layer / layers[-1]) * 100 if layers[-1] > 0 else 0.0

        results.append({
            "model": model_short_name,
            "threshold": threshold,
            "accuracy": accuracy,
            "avg_exit_layer": avg_exit_layer,
            "layers_saved_pct": layers_saved_pct,
            "n_samples": n_samples,
        })
        print(f"  Threshold {threshold:.2f}: accuracy={accuracy:.4f}, "
              f"layers saved={layers_saved_pct:.1f}%")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved early exit results to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate early exit with probes")
    parser.add_argument("--hidden_states_dir", type=str, required=True)
    parser.add_argument("--probes_dir", type=str, required=True)
    parser.add_argument("--model_short_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    simulate_early_exit(
        hidden_states_dir=args.hidden_states_dir,
        probes_dir=args.probes_dir,
        model_short_name=args.model_short_name,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
