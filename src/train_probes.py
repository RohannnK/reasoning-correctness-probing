"""Train linear and MLP probes on hidden states to detect correctness signals.

Includes position-controlled validation: all-chunks (position confounded),
last-chunk-only (position controlled), and position-only baseline analyses.
"""

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

from src.utils import compute_ece, load_jsonl

DEFAULT_HIDDEN_STATES_DIR = "data/hidden_states"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_CHUNKS_PATH = "data/chunks/gsm8k_chunks.jsonl"

MODEL_KEYS = ["base", "distilled"]
MODEL_LABELS = {
    "base": "Base (Qwen2.5-Math-1.5B)",
    "distilled": "Distilled (R1-Distill-Qwen-1.5B)",
}
C_VALUES = [0.01, 0.1, 1.0, 10.0]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _load_layer_data(
    model_dir: str, problems: list[dict],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load hidden states with per-sample position metadata.

    Args:
        model_dir: Directory with problem_*.npz files for one model.
        problems: Loaded chunks JSONL records.

    Returns:
        Dict mapping layer_idx to (X, y, positions, is_last) arrays.
    """
    layer_vecs: dict[int, list[np.ndarray]] = {}
    layer_labels: dict[int, list[int]] = {}
    layer_positions: dict[int, list[float]] = {}
    layer_is_last: dict[int, list[bool]] = {}

    for problem in problems:
        pid = problem["problem_id"]
        npz_path = os.path.join(model_dir, f"problem_{pid}.npz")
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path)
        labels = data["labels"]
        num_chunks = len(labels)

        for key in data.files:
            if key == "labels":
                continue
            layer_idx = int(key.replace("layer_", ""))
            if layer_idx not in layer_vecs:
                layer_vecs[layer_idx] = []
                layer_labels[layer_idx] = []
                layer_positions[layer_idx] = []
                layer_is_last[layer_idx] = []

            vecs = data[key]  # (num_chunks, hidden_dim)
            for i in range(num_chunks):
                layer_vecs[layer_idx].append(vecs[i])
                layer_labels[layer_idx].append(int(labels[i]))
                layer_positions[layer_idx].append(i / max(num_chunks - 1, 1))
                layer_is_last[layer_idx].append(i == num_chunks - 1)

    result = {}
    for layer_idx in sorted(layer_vecs.keys()):
        result[layer_idx] = (
            np.array(layer_vecs[layer_idx]),
            np.array(layer_labels[layer_idx]),
            np.array(layer_positions[layer_idx]),
            np.array(layer_is_last[layer_idx]),
        )
    return result


def _eval_probe(probe, X_test, y_test):
    """Evaluate a fitted probe, returning metrics dict."""
    probs = probe.predict_proba(X_test)[:, 1]
    preds = probe.predict(X_test)
    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "ece": compute_ece(y_test, probs),
    }


def _train_linear(X_train, y_train, X_test, y_test):
    """Train LR with grid search, return (metrics_dict, best_C)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    lr_cv = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"),
        param_grid={"C": C_VALUES},
        scoring="roc_auc",
        cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    lr_cv.fit(X_tr, y_train)
    metrics = _eval_probe(lr_cv.best_estimator_, X_te, y_test)
    return metrics, lr_cv.best_params_["C"]


def _train_mlp(X_train, y_train, X_test, y_test):
    """Train MLP with sample weights, return metrics_dict."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    sample_weights = compute_sample_weight("balanced", y_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256,),
        early_stopping=True,
        max_iter=500,
        random_state=RANDOM_STATE,
        validation_fraction=0.15,
    )
    mlp.fit(X_tr, y_train, sample_weight=sample_weights)
    return _eval_probe(mlp, X_te, y_test)


def train_probes(
    hidden_states_dir: str, output_dir: str, chunks_path: str,
) -> pd.DataFrame:
    """Train probes with position-controlled validation.

    Runs three analysis types:
    - all_chunks: original (position confounded)
    - last_chunk_only: position controlled (only final chunks)
    - position_only_baseline: LR on relative position alone

    Args:
        hidden_states_dir: Root directory containing base/ and distilled/ subdirs.
        output_dir: Root output directory for metrics and figures.
        chunks_path: Path to gsm8k_chunks.jsonl.

    Returns:
        DataFrame with all probe results.
    """
    print(f"Loading chunks from {chunks_path}...")
    problems = load_jsonl(chunks_path)
    print(f"Loaded {len(problems)} problems")

    results = []
    position_baseline_done = False

    for model_key in MODEL_KEYS:
        model_dir = os.path.join(hidden_states_dir, model_key)
        print(f"\n{'='*60}")
        print(f"Loading hidden states for {model_key}...")
        layer_data = _load_layer_data(model_dir, problems)

        if not layer_data:
            print(f"No hidden states found in {model_dir}")
            continue

        print(f"Loaded {len(layer_data)} layers")

        for layer_idx, (X, y, positions, is_last) in sorted(layer_data.items()):
            print(f"\n[{model_key}] Layer {layer_idx}: {X.shape[0]} samples, "
                  f"positive rate: {y.mean():.3f}")

            if len(np.unique(y)) < 2:
                print(f"  Skipping: only one class present.")
                continue

            # ---- Position-only baseline (fit once, duplicate for both models) ----
            if not position_baseline_done:
                pos_train, pos_test, yp_train, yp_test = train_test_split(
                    positions.reshape(-1, 1), y,
                    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
                )
                pos_lr = LogisticRegression(
                    class_weight="balanced", max_iter=1000, solver="lbfgs",
                )
                pos_lr.fit(pos_train, yp_train)
                pos_metrics = _eval_probe(pos_lr, pos_test, yp_test)

                for mk in MODEL_KEYS:
                    results.append({
                        "model_key": mk,
                        "layer": layer_idx,
                        "probe_type": "linear",
                        "analysis_type": "position_only_baseline",
                        "best_C": None,
                        **pos_metrics,
                    })

                print(f"  Position-only baseline AUC={pos_metrics['roc_auc']:.4f}")
                position_baseline_done = True

            # ---- All-chunks probes (position confounded) ----
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
            )

            lr_metrics, best_c = _train_linear(X_train, y_train, X_test, y_test)
            results.append({
                "model_key": model_key, "layer": layer_idx,
                "probe_type": "linear", "analysis_type": "all_chunks",
                "best_C": best_c, **lr_metrics,
            })

            mlp_metrics = _train_mlp(X_train, y_train, X_test, y_test)
            results.append({
                "model_key": model_key, "layer": layer_idx,
                "probe_type": "mlp", "analysis_type": "all_chunks",
                "best_C": None, **mlp_metrics,
            })

            print(f"  All-chunks  Linear AUC={lr_metrics['roc_auc']:.4f}  "
                  f"MLP AUC={mlp_metrics['roc_auc']:.4f}")

            # ---- Last-chunk-only probes (position controlled) ----
            mask = is_last
            X_last, y_last = X[mask], y[mask]

            if len(np.unique(y_last)) < 2:
                print(f"  Last-chunk-only: only one class, skipping.")
                continue

            X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
                X_last, y_last,
                test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_last,
            )

            lr_last_metrics, best_c_last = _train_linear(
                X_train_l, y_train_l, X_test_l, y_test_l,
            )
            results.append({
                "model_key": model_key, "layer": layer_idx,
                "probe_type": "linear", "analysis_type": "last_chunk_only",
                "best_C": best_c_last, **lr_last_metrics,
            })

            mlp_last_metrics = _train_mlp(
                X_train_l, y_train_l, X_test_l, y_test_l,
            )
            results.append({
                "model_key": model_key, "layer": layer_idx,
                "probe_type": "mlp", "analysis_type": "last_chunk_only",
                "best_C": None, **mlp_last_metrics,
            })

            print(f"  Last-chunk  Linear AUC={lr_last_metrics['roc_auc']:.4f}  "
                  f"MLP AUC={mlp_last_metrics['roc_auc']:.4f}")

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
    """Generate 1x3 layer-wise ROC-AUC comparison plot."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Get position-only baseline AUC for the reference line
    pos_df = df[df["analysis_type"] == "position_only_baseline"]
    pos_auc = pos_df.iloc[0]["roc_auc"] if not pos_df.empty else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    subplot_configs = [
        (axes[0], "linear", "all_chunks",
         "Linear — All Chunks (position confounded)"),
        (axes[1], "linear", "last_chunk_only",
         "Linear — Last Chunk Only (position controlled)"),
        (axes[2], "mlp", "last_chunk_only",
         "MLP — Last Chunk Only (position controlled)"),
    ]

    for ax, probe_type, analysis_type, title in subplot_configs:
        for model_key in MODEL_KEYS:
            subset = df[
                (df["model_key"] == model_key)
                & (df["probe_type"] == probe_type)
                & (df["analysis_type"] == analysis_type)
            ].sort_values("layer")

            if subset.empty:
                continue

            ax.plot(
                subset["layer"], subset["roc_auc"],
                marker="o", markersize=3, label=MODEL_LABELS[model_key],
            )

        if pos_auc is not None:
            ax.axhline(y=pos_auc, color="red", linestyle="--", linewidth=1,
                        label=f"Position-only baseline ({pos_auc:.2f})")
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1,
                    label="Random baseline")
        ax.set_xlabel("Layer")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Layer-wise Probe ROC-AUC: Base vs Distilled", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(figures_dir, "layer_wise_roc_auc.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _print_summary(df: pd.DataFrame) -> None:
    """Print best layer and ROC-AUC for each (model, probe_type, analysis_type)."""
    print(f"\n{'='*60}")
    print("Summary: Best layer per (model, probe_type, analysis_type)")
    print(f"{'='*60}")

    # Position-only baseline (single value)
    pos_df = df[df["analysis_type"] == "position_only_baseline"]
    if not pos_df.empty:
        pos_auc = pos_df.iloc[0]["roc_auc"]
        print(f"\n  Position-only baseline AUC: {pos_auc:.4f}")

    for analysis_type in ["all_chunks", "last_chunk_only"]:
        print(f"\n  --- {analysis_type} ---")
        for model_key in MODEL_KEYS:
            for probe_type in ["linear", "mlp"]:
                subset = df[
                    (df["model_key"] == model_key)
                    & (df["probe_type"] == probe_type)
                    & (df["analysis_type"] == analysis_type)
                ]
                if subset.empty:
                    continue
                best = subset.loc[subset["roc_auc"].idxmax()]
                print(f"  {model_key:>10} / {probe_type:<6}  →  "
                      f"layer {int(best['layer']):2d}  ROC-AUC={best['roc_auc']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train probes with position-controlled validation",
    )
    parser.add_argument(
        "--hidden_states_dir", type=str, default=DEFAULT_HIDDEN_STATES_DIR,
        help=f"Root dir with base/ and distilled/ subdirs (default: {DEFAULT_HIDDEN_STATES_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir for metrics/ and figures/ (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunks_path", type=str, default=DEFAULT_CHUNKS_PATH,
        help=f"Path to chunks JSONL (default: {DEFAULT_CHUNKS_PATH})",
    )
    args = parser.parse_args()

    train_probes(
        hidden_states_dir=args.hidden_states_dir,
        output_dir=args.output_dir,
        chunks_path=args.chunks_path,
    )


if __name__ == "__main__":
    main()
