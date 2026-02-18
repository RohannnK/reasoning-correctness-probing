"""Simulate chunk-level early exit using probe confidence on the distilled model."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.utils import load_jsonl

DEFAULT_HIDDEN_STATES_DIR = "data/hidden_states"
DEFAULT_CHUNKS_PATH = "data/chunks/gsm8k_chunks.jsonl"
DEFAULT_PROBE_RESULTS_PATH = "results/metrics/probe_results.csv"
DEFAULT_OUTPUT_DIR = "results"

C_VALUES = [0.01, 0.1, 1.0, 10.0]
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def _get_best_layer(probe_results_path: str) -> int:
    """Find the best layer by average ROC-AUC across models (last_chunk_only, linear)."""
    df = pd.read_csv(probe_results_path)
    subset = df[
        (df["analysis_type"] == "last_chunk_only")
        & (df["probe_type"] == "linear")
    ]
    avg_auc = subset.groupby("layer")["roc_auc"].mean()
    best_layer = int(avg_auc.idxmax())
    print(f"  Best layer: {best_layer} (avg AUC={avg_auc[best_layer]:.4f})")
    return best_layer


def _load_distilled_layer(
    hidden_states_dir: str, problems: list[dict], best_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load distilled model hidden states at best layer for probe training.

    Returns:
        (X, y, is_last, problem_ids) — all chunks, flattened across problems.
    """
    dist_dir = os.path.join(hidden_states_dir, "distilled")
    layer_key = f"layer_{best_layer}"

    vecs, labels, is_last, pids = [], [], [], []
    for problem in problems:
        pid = problem["problem_id"]
        npz_path = os.path.join(dist_dir, f"problem_{pid}.npz")
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path)
        chunk_labels = data["labels"]
        num_chunks = len(chunk_labels)
        chunk_vecs = data[layer_key]

        for i in range(num_chunks):
            vecs.append(chunk_vecs[i])
            labels.append(int(chunk_labels[i]))
            is_last.append(i == num_chunks - 1)
            pids.append(pid)

    return np.array(vecs), np.array(labels), np.array(is_last), np.array(pids)


def _train_exit_probe(
    X, y, is_last, problem_ids,
) -> tuple:
    """Train a probe on last-chunk-only data, return (probe, scaler, test_pids).

    Matches Day 6 train_probes._train_linear() configuration exactly.
    """
    mask = is_last
    X_last = X[mask]
    y_last = y[mask]
    pids_last = problem_ids[mask]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_last, y_last, groups=pids_last))

    X_train, y_train = X_last[train_idx], y_last[train_idx]
    X_test, y_test = X_last[test_idx], y_last[test_idx]
    test_pids = set(pids_last[test_idx].tolist())

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

    probe = lr_cv.best_estimator_
    probs = probe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"  Probe trained: best_C={lr_cv.best_params_['C']}, "
          f"test AUC={auc:.4f}, test problems={len(test_pids)}")

    return probe, scaler, test_pids


def _simulate_early_exit(
    problems: list[dict], hidden_states_dir: str,
    probe, scaler, best_layer: int, test_pids: set,
) -> pd.DataFrame:
    """Run early exit simulation on test-set problems for the distilled model."""
    dist_dir = os.path.join(hidden_states_dir, "distilled")
    layer_key = f"layer_{best_layer}"

    # Build problem lookup
    problem_map = {p["problem_id"]: p for p in problems}

    # Collect per-problem simulation data
    problem_results = []
    for pid in sorted(test_pids):
        if pid not in problem_map:
            continue
        problem = problem_map[pid]
        chunks = problem["chunks"]
        num_chunks = len(chunks)

        npz_path = os.path.join(dist_dir, f"problem_{pid}.npz")
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path)
        hidden_vecs = data[layer_key]  # (num_chunks, 1536)
        total_chars = chunks[-1]["char_end"]

        # Get probe probabilities for ALL chunks
        X_scaled = scaler.transform(hidden_vecs)
        probs = probe.predict_proba(X_scaled)[:, 1]

        problem_results.append({
            "pid": pid,
            "num_chunks": num_chunks,
            "total_chars": total_chars,
            "chunk_probs": probs,
            "chunk_is_correct": [c["is_correct"] for c in chunks],
            "chunk_char_end": [c["char_end"] for c in chunks],
        })

    # Simulate each threshold
    results = []

    # "No exit" baseline
    baseline_correct = sum(
        p["chunk_is_correct"][-1] for p in problem_results
    )
    baseline_accuracy = baseline_correct / len(problem_results)
    results.append({
        "threshold": 1.0,
        "accuracy": baseline_accuracy,
        "mean_tokens_saved_pct": 0.0,
        "median_tokens_saved_pct": 0.0,
        "num_early_exits": 0,
        "num_problems": len(problem_results),
        "label": "no_exit",
    })

    for tau in THRESHOLDS:
        exit_correct = 0
        tokens_saved = []
        early_exits = 0

        for pr in problem_results:
            exited = False
            for i in range(pr["num_chunks"]):
                if pr["chunk_probs"][i] > tau:
                    # Exit at this chunk
                    exit_correct += int(pr["chunk_is_correct"][i])
                    chars_used = pr["chunk_char_end"][i]
                    tokens_saved.append(1.0 - chars_used / pr["total_chars"])
                    if i < pr["num_chunks"] - 1:
                        early_exits += 1
                    exited = True
                    break

            if not exited:
                # Default: use final chunk
                exit_correct += int(pr["chunk_is_correct"][-1])
                tokens_saved.append(0.0)

        accuracy = exit_correct / len(problem_results)
        results.append({
            "threshold": tau,
            "accuracy": accuracy,
            "mean_tokens_saved_pct": float(np.mean(tokens_saved) * 100),
            "median_tokens_saved_pct": float(np.median(tokens_saved) * 100),
            "num_early_exits": early_exits,
            "num_problems": len(problem_results),
            "label": f"τ={tau}",
        })

    return pd.DataFrame(results)


def _plot_pareto(df: pd.DataFrame, output_dir: str) -> None:
    """Pareto frontier: mean tokens saved vs accuracy."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Threshold points (exclude no_exit baseline from main curve)
    thresh_df = df[df["label"] != "no_exit"].sort_values("mean_tokens_saved_pct")
    ax.plot(
        thresh_df["mean_tokens_saved_pct"], thresh_df["accuracy"],
        marker="o", markersize=6, linewidth=1.5, color="#1f77b4",
        label="Early exit (distilled)",
    )

    # Annotate each point with threshold
    for _, row in thresh_df.iterrows():
        ax.annotate(
            row["label"],
            (row["mean_tokens_saved_pct"], row["accuracy"]),
            textcoords="offset points", xytext=(8, 4), fontsize=8,
        )

    # No-exit baseline
    baseline = df[df["label"] == "no_exit"].iloc[0]
    ax.scatter(
        [baseline["mean_tokens_saved_pct"]], [baseline["accuracy"]],
        marker="*", s=150, color="red", zorder=5,
        label=f"No exit (acc={baseline['accuracy']:.3f})",
    )

    ax.set_xlabel("Mean Tokens Saved (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Early Exit Pareto Frontier (Distilled Model, Post-hoc Simulation)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(figures_dir, "early_exit_pareto.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _print_summary(df: pd.DataFrame) -> None:
    """Print summary table of early exit results."""
    print(f"\n{'='*60}")
    print("Early Exit Simulation Summary (Distilled Model)")
    print(f"{'='*60}")
    print("\n  NOTE: Probe trained on last-chunk-only data but applied to ALL chunks.")
    print("  This is a post-hoc simulation, not real-time inference.")
    print("  'Accuracy' = fraction of problems where the exit chunk's intermediate")
    print("  answer is_correct.\n")

    print(f"  {'Threshold':>10}  {'Accuracy':>9}  {'Tokens Saved':>13}  "
          f"{'Median Saved':>13}  {'Early Exits':>12}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*13}  {'-'*13}  {'-'*12}")

    for _, row in df.iterrows():
        label = row["label"]
        print(f"  {label:>10}  {row['accuracy']:9.3f}  "
              f"{row['mean_tokens_saved_pct']:12.1f}%  "
              f"{row['median_tokens_saved_pct']:12.1f}%  "
              f"{int(row['num_early_exits']):>5} / {int(row['num_problems'])}")


def early_exit(
    hidden_states_dir: str, chunks_path: str,
    probe_results_path: str, output_dir: str,
) -> pd.DataFrame:
    """Run early exit simulation."""
    print(f"Loading chunks from {chunks_path}...")
    problems = load_jsonl(chunks_path)
    print(f"Loaded {len(problems)} problems")

    print("\nSelecting best layer...")
    best_layer = _get_best_layer(probe_results_path)

    print(f"\nLoading distilled hidden states at layer {best_layer}...")
    X, y, is_last, pids = _load_distilled_layer(hidden_states_dir, problems, best_layer)
    print(f"Loaded {len(X)} samples ({is_last.sum()} last-chunk)")

    print("\nTraining exit probe (last-chunk-only, GridSearchCV)...")
    probe, scaler, test_pids = _train_exit_probe(X, y, is_last, pids)

    print(f"\nSimulating early exit on {len(test_pids)} test problems...")
    results_df = _simulate_early_exit(
        problems, hidden_states_dir, probe, scaler, best_layer, test_pids,
    )

    # Save CSV
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, "early_exit_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    _plot_pareto(results_df, output_dir)
    _print_summary(results_df)

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate chunk-level early exit with probe confidence",
    )
    parser.add_argument(
        "--hidden_states_dir", type=str, default=DEFAULT_HIDDEN_STATES_DIR,
        help=f"Root dir with base/ and distilled/ subdirs (default: {DEFAULT_HIDDEN_STATES_DIR})",
    )
    parser.add_argument(
        "--chunks_path", type=str, default=DEFAULT_CHUNKS_PATH,
        help=f"Path to chunks JSONL (default: {DEFAULT_CHUNKS_PATH})",
    )
    parser.add_argument(
        "--probe_results_path", type=str, default=DEFAULT_PROBE_RESULTS_PATH,
        help=f"Path to probe results CSV (default: {DEFAULT_PROBE_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir for metrics/ and figures/ (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    early_exit(
        hidden_states_dir=args.hidden_states_dir,
        chunks_path=args.chunks_path,
        probe_results_path=args.probe_results_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
