"""Train linear and MLP probes on hidden states to detect correctness signals."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.utils import compute_ece, load_hidden_states

DEFAULT_HIDDEN_STATES_DIR = "data/hidden_states"
DEFAULT_OUTPUT_DIR = "results"

MODEL_KEYS = ["base", "distilled"]
MODEL_LABELS = {
    "base": "Base (Qwen2.5-Math-1.5B)",
    "distilled": "Distilled (R1-Distill-Qwen-1.5B)",
}
C_VALUES = [0.01, 0.1, 1.0, 10.0]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_probes(hidden_states_dir: str, output_dir: str) -> pd.DataFrame:
    """Train LogisticRegression and MLP probes for each model and layer.

    Args:
        hidden_states_dir: Root directory containing base/ and distilled/ subdirs.
        output_dir: Root output directory for metrics and figures.

    Returns:
        DataFrame with probe results.
    """
    results = []

    for model_key in MODEL_KEYS:
        model_dir = os.path.join(hidden_states_dir, model_key)
        print(f"\n{'='*60}")
        print(f"Loading hidden states for {model_key}...")
        layer_data = load_hidden_states(model_dir)

        if not layer_data:
            print(f"No hidden states found in {model_dir}")
            continue

        print(f"Loaded {len(layer_data)} layers")

        for layer_idx, (X, y) in sorted(layer_data.items()):
            print(f"\n[{model_key}] Layer {layer_idx}: {X.shape[0]} samples, "
                  f"{X.shape[1]} features, positive rate: {y.mean():.3f}")

            if len(np.unique(y)) < 2:
                print(f"  Skipping: only one class present.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
            )

            # Fit scaler on training data only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # --- Logistic Regression with grid search ---
            lr_cv = GridSearchCV(
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, solver="lbfgs",
                ),
                param_grid={"C": C_VALUES},
                scoring="roc_auc",
                cv=StratifiedShuffleSplit(
                    n_splits=3, test_size=0.2, random_state=RANDOM_STATE,
                ),
                n_jobs=-1,
            )
            lr_cv.fit(X_train_scaled, y_train)
            lr_best = lr_cv.best_estimator_

            lr_probs = lr_best.predict_proba(X_test_scaled)[:, 1]
            lr_preds = lr_best.predict(X_test_scaled)

            results.append({
                "model_key": model_key,
                "layer": layer_idx,
                "probe_type": "linear",
                "best_C": lr_cv.best_params_["C"],
                "roc_auc": roc_auc_score(y_test, lr_probs),
                "precision": precision_score(y_test, lr_preds, zero_division=0),
                "recall": recall_score(y_test, lr_preds, zero_division=0),
                "f1": f1_score(y_test, lr_preds, zero_division=0),
                "ece": compute_ece(y_test, lr_probs),
            })

            # --- MLP probe with sample weights for class imbalance ---
            sample_weights = compute_sample_weight("balanced", y_train)

            mlp = MLPClassifier(
                hidden_layer_sizes=(256,),
                early_stopping=True,
                max_iter=500,
                random_state=RANDOM_STATE,
                validation_fraction=0.15,
            )
            mlp.fit(X_train_scaled, y_train, sample_weight=sample_weights)

            mlp_probs = mlp.predict_proba(X_test_scaled)[:, 1]
            mlp_preds = mlp.predict(X_test_scaled)

            results.append({
                "model_key": model_key,
                "layer": layer_idx,
                "probe_type": "mlp",
                "best_C": None,
                "roc_auc": roc_auc_score(y_test, mlp_probs),
                "precision": precision_score(y_test, mlp_preds, zero_division=0),
                "recall": recall_score(y_test, mlp_preds, zero_division=0),
                "f1": f1_score(y_test, mlp_preds, zero_division=0),
                "ece": compute_ece(y_test, mlp_probs),
            })

            print(f"  Linear AUC={results[-2]['roc_auc']:.4f}  F1={results[-2]['f1']:.4f}  "
                  f"best_C={lr_cv.best_params_['C']}")
            print(f"  MLP    AUC={results[-1]['roc_auc']:.4f}  F1={results[-1]['f1']:.4f}")

    df = pd.DataFrame(results)

    # Save CSV
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, "probe_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Generate figure
    _plot_layer_roc_auc(df, output_dir)

    # Print summary
    _print_summary(df)

    return df


def _plot_layer_roc_auc(df: pd.DataFrame, output_dir: str) -> None:
    """Generate layer-wise ROC-AUC comparison plot."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    probe_types = [("linear", "Linear Probe"), ("mlp", "MLP Probe")]

    for ax, (ptype, title) in zip(axes, probe_types):
        for model_key in MODEL_KEYS:
            subset = df[(df["model_key"] == model_key) & (df["probe_type"] == ptype)]
            subset = subset.sort_values("layer")
            ax.plot(
                subset["layer"], subset["roc_auc"],
                marker="o", markersize=3, label=MODEL_LABELS[model_key],
            )

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline")
        ax.set_xlabel("Layer")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Layer-wise Probe ROC-AUC: Base vs Distilled", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(figures_dir, "layer_wise_roc_auc.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _print_summary(df: pd.DataFrame) -> None:
    """Print best layer and ROC-AUC for each (model, probe_type)."""
    print(f"\n{'='*60}")
    print("Summary: Best layer per (model, probe_type)")
    print(f"{'='*60}")

    for model_key in MODEL_KEYS:
        for probe_type in ["linear", "mlp"]:
            subset = df[(df["model_key"] == model_key) & (df["probe_type"] == probe_type)]
            if subset.empty:
                continue
            best = subset.loc[subset["roc_auc"].idxmax()]
            print(f"  {model_key:>10} / {probe_type:<6}  →  "
                  f"layer {int(best['layer']):2d}  ROC-AUC={best['roc_auc']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes on hidden states")
    parser.add_argument(
        "--hidden_states_dir", type=str, default=DEFAULT_HIDDEN_STATES_DIR,
        help=f"Root dir with base/ and distilled/ subdirs (default: {DEFAULT_HIDDEN_STATES_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir for metrics/ and figures/ (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    train_probes(
        hidden_states_dir=args.hidden_states_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
