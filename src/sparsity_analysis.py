"""L1-regularized probe sparsity analysis with dimension overlap and weight correlation."""

import argparse
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.utils import load_jsonl

DEFAULT_HIDDEN_STATES_DIR = "data/hidden_states"
DEFAULT_CHUNKS_PATH = "data/chunks/gsm8k_chunks.jsonl"
DEFAULT_PROBE_RESULTS_PATH = "results/metrics/probe_results.csv"
DEFAULT_OUTPUT_DIR = "results"

C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
FIXED_C = 0.1
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K = 20
MODEL_KEYS = ["base", "distilled"]
MODEL_LABELS = {
    "base": "Base (Qwen2.5-Math-1.5B)",
    "distilled": "Distilled (R1-Distill-Qwen-1.5B)",
}


def _load_aligned_data(
    hidden_states_dir: str, problems: list[dict],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load aligned hidden states for both base and distilled models.

    Returns:
        Dict mapping layer_idx to (X_base, X_dist, labels, is_last, problem_ids).
    """
    base_dir = os.path.join(hidden_states_dir, "base")
    dist_dir = os.path.join(hidden_states_dir, "distilled")

    layer_base: dict[int, list[np.ndarray]] = {}
    layer_dist: dict[int, list[np.ndarray]] = {}
    layer_labels: dict[int, list[int]] = {}
    layer_is_last: dict[int, list[bool]] = {}
    layer_pids: dict[int, list[int]] = {}

    skipped = 0
    for problem in problems:
        pid = problem["problem_id"]
        base_path = os.path.join(base_dir, f"problem_{pid}.npz")
        dist_path = os.path.join(dist_dir, f"problem_{pid}.npz")

        if not os.path.exists(base_path) or not os.path.exists(dist_path):
            skipped += 1
            continue

        base_data = np.load(base_path)
        dist_data = np.load(dist_path)

        base_labels = base_data["labels"]
        dist_labels = dist_data["labels"]
        assert len(base_labels) == len(dist_labels), (
            f"Problem {pid}: chunk count mismatch "
            f"(base={len(base_labels)}, distilled={len(dist_labels)})"
        )
        num_chunks = len(base_labels)

        for key in base_data.files:
            if key == "labels":
                continue
            layer_idx = int(key.replace("layer_", ""))
            if layer_idx not in layer_base:
                layer_base[layer_idx] = []
                layer_dist[layer_idx] = []
                layer_labels[layer_idx] = []
                layer_is_last[layer_idx] = []
                layer_pids[layer_idx] = []

            base_vecs = base_data[key]
            dist_vecs = dist_data[key]
            for i in range(num_chunks):
                layer_base[layer_idx].append(base_vecs[i])
                layer_dist[layer_idx].append(dist_vecs[i])
                layer_labels[layer_idx].append(int(base_labels[i]))
                layer_is_last[layer_idx].append(i == num_chunks - 1)
                layer_pids[layer_idx].append(pid)

    if skipped:
        print(f"  Skipped {skipped} problems (missing one or both npz files)")

    result = {}
    for layer_idx in sorted(layer_base.keys()):
        result[layer_idx] = (
            np.array(layer_base[layer_idx]),
            np.array(layer_dist[layer_idx]),
            np.array(layer_labels[layer_idx]),
            np.array(layer_is_last[layer_idx]),
            np.array(layer_pids[layer_idx]),
        )
    return result


def _get_best_layer(probe_results_path: str) -> int:
    """Find the best layer by average ROC-AUC across models (last_chunk_only, linear)."""
    df = pd.read_csv(probe_results_path)
    subset = df[
        (df["analysis_type"] == "last_chunk_only")
        & (df["probe_type"] == "linear")
    ]
    avg_auc = subset.groupby("layer")["roc_auc"].mean()
    best_layer = int(avg_auc.idxmax())
    print(f"  Best layer (avg last-chunk-only linear AUC): {best_layer} "
          f"(AUC={avg_auc[best_layer]:.4f})")
    return best_layer


def _train_l1_probe(X_train, y_train, X_test, y_test, C):
    """Train an L1 probe at a given C and return (roc_auc, num_nonzero, coefs)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    probe = LogisticRegression(
        solver="saga", C=C, l1_ratio=1, class_weight="balanced",
        max_iter=5000, random_state=RANDOM_STATE,
    )
    probe.fit(X_tr, y_train)

    probs = probe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, probs)
    coefs = probe.coef_[0]
    num_nonzero = int(np.count_nonzero(coefs))

    return auc, num_nonzero, coefs


def _core_sparsity(layer_data: dict, output_dir: str) -> pd.DataFrame:
    """Run L1 grid search for both models across all layers (last-chunk only)."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    results = []

    for layer_idx in sorted(layer_data.keys()):
        X_base, X_dist, labels, is_last, pids = layer_data[layer_idx]

        # Filter to last chunks
        mask = is_last
        y = labels[mask]
        g = pids[mask]

        if len(np.unique(y)) < 2:
            print(f"  Layer {layer_idx}: skipping (single class)")
            continue

        train_idx, test_idx = next(gss.split(y, y, groups=g))
        y_train, y_test = y[train_idx], y[test_idx]

        for model_key, X_all in [("base", X_base), ("distilled", X_dist)]:
            X = X_all[mask]
            X_train, X_test = X[train_idx], X[test_idx]

            best_auc = -1
            best_row = None
            for C in C_VALUES:
                auc, num_nonzero, coefs = _train_l1_probe(
                    X_train, y_train, X_test, y_test, C,
                )
                if auc > best_auc:
                    best_auc = auc
                    top_indices = np.argsort(np.abs(coefs))[-TOP_K:][::-1].tolist()
                    best_row = {
                        "model": model_key,
                        "layer": layer_idx,
                        "best_C": C,
                        "roc_auc": auc,
                        "num_nonzero": num_nonzero,
                        "sparsity_ratio": num_nonzero / len(coefs),
                        "top_20_dims": str(top_indices),
                    }

            results.append(best_row)

        print(f"  Layer {layer_idx:2d}: "
              f"base={results[-2]['num_nonzero']:4d} dims (AUC={results[-2]['roc_auc']:.4f})  "
              f"dist={results[-1]['num_nonzero']:4d} dims (AUC={results[-1]['roc_auc']:.4f})")

    df = pd.DataFrame(results)
    csv_path = os.path.join(metrics_dir, "sparsity_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved sparsity results to {csv_path}")
    return df


def _fixed_c_sparsity(layer_data: dict, output_dir: str) -> pd.DataFrame:
    """Run L1 probes at fixed C=0.1 across all layers for smooth sparsity curves."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    results = []

    for layer_idx in sorted(layer_data.keys()):
        X_base, X_dist, labels, is_last, pids = layer_data[layer_idx]

        mask = is_last
        y = labels[mask]
        g = pids[mask]

        if len(np.unique(y)) < 2:
            print(f"  Layer {layer_idx}: skipping (single class)")
            continue

        train_idx, test_idx = next(gss.split(y, y, groups=g))
        y_train, y_test = y[train_idx], y[test_idx]

        for model_key, X_all in [("base", X_base), ("distilled", X_dist)]:
            X = X_all[mask]
            X_train, X_test = X[train_idx], X[test_idx]

            auc, num_nonzero, _ = _train_l1_probe(
                X_train, y_train, X_test, y_test, FIXED_C,
            )
            results.append({
                "model": model_key,
                "layer": layer_idx,
                "C": FIXED_C,
                "roc_auc": auc,
                "num_nonzero": num_nonzero,
                "sparsity_ratio": num_nonzero / 1536,
            })

        print(f"  Layer {layer_idx:2d}: "
              f"base={results[-2]['num_nonzero']:3d} dims (AUC={results[-2]['roc_auc']:.4f})  "
              f"dist={results[-1]['num_nonzero']:3d} dims (AUC={results[-1]['roc_auc']:.4f})")

    df = pd.DataFrame(results)
    csv_path = os.path.join(metrics_dir, "sparsity_results_fixed_c.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved fixed-C sparsity results to {csv_path}")
    return df


def _sparsity_tradeoff(
    layer_data: dict, best_layer: int, output_dir: str,
) -> pd.DataFrame:
    """Sweep C values at the best layer with a single fixed split."""
    metrics_dir = os.path.join(output_dir, "metrics")

    X_base, X_dist, labels, is_last, pids = layer_data[best_layer]
    mask = is_last
    y = labels[mask]
    g = pids[mask]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(y, y, groups=g))
    y_train, y_test = y[train_idx], y[test_idx]

    rows = []
    for model_key, X_all in [("base", X_base), ("distilled", X_dist)]:
        X = X_all[mask]
        X_train, X_test = X[train_idx], X[test_idx]

        for C in C_VALUES:
            auc, num_nonzero, _ = _train_l1_probe(X_train, y_train, X_test, y_test, C)
            rows.append({
                "model": model_key,
                "layer": best_layer,
                "C": C,
                "roc_auc": auc,
                "num_nonzero": num_nonzero,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(metrics_dir, "sparsity_tradeoff.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved sparsity tradeoff to {csv_path}")
    return df


def _dimension_overlap(
    layer_data: dict, best_layer: int, output_dir: str,
) -> dict:
    """Compute dimension overlap and weight correlation at the best layer."""
    metrics_dir = os.path.join(output_dir, "metrics")

    X_base, X_dist, labels, is_last, pids = layer_data[best_layer]
    mask = is_last
    y = labels[mask]
    g = pids[mask]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(y, y, groups=g))
    y_train, y_test = y[train_idx], y[test_idx]

    # Train at fixed C=0.1 for meaningful sparsity comparison
    model_coefs = {}
    for model_key, X_all in [("base", X_base), ("distilled", X_dist)]:
        X = X_all[mask]
        X_train, X_test = X[train_idx], X[test_idx]

        _, _, coefs = _train_l1_probe(X_train, y_train, X_test, y_test, FIXED_C)
        model_coefs[model_key] = coefs

    base_nonzero = set(np.nonzero(model_coefs["base"])[0].tolist())
    dist_nonzero = set(np.nonzero(model_coefs["distilled"])[0].tolist())

    shared = base_nonzero & dist_nonzero
    union = base_nonzero | dist_nonzero
    jaccard = len(shared) / len(union) if union else 0.0

    # Weight correlation on shared dimensions
    weight_corr = float("nan")
    if len(shared) >= 2:
        shared_idx = sorted(shared)
        w_base = model_coefs["base"][shared_idx]
        w_dist = model_coefs["distilled"][shared_idx]
        weight_corr, _ = pearsonr(w_base, w_dist)
        weight_corr = float(weight_corr)

    overlap_info = {
        "best_layer": best_layer,
        "C": FIXED_C,
        "base_nonzero_count": len(base_nonzero),
        "distilled_nonzero_count": len(dist_nonzero),
        "shared_count": len(shared),
        "base_only_count": len(base_nonzero - dist_nonzero),
        "distilled_only_count": len(dist_nonzero - base_nonzero),
        "jaccard_similarity": jaccard,
        "weight_correlation_shared_dims": weight_corr,
    }

    json_path = os.path.join(metrics_dir, "dimension_overlap.json")
    with open(json_path, "w") as f:
        json.dump(overlap_info, f, indent=2)
    print(f"Saved dimension overlap to {json_path}")
    return overlap_info


def _plot_sparsity_by_layer(fixed_c_df: pd.DataFrame, output_dir: str) -> None:
    """Plot nonzero dimensions and ROC-AUC at fixed C across layers."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_key in MODEL_KEYS:
        subset = fixed_c_df[fixed_c_df["model"] == model_key].sort_values("layer")
        ax1.plot(subset["layer"], subset["num_nonzero"],
                 marker="o", markersize=3, label=MODEL_LABELS[model_key])
        ax2.plot(subset["layer"], subset["roc_auc"],
                 marker="o", markersize=3, label=MODEL_LABELS[model_key])

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Number of Nonzero Dimensions (out of 1536)")
    ax1.set_title(f"L1 Probe Sparsity (C={FIXED_C}, Last-Chunk Only)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title(f"Sparse Probe Accuracy (C={FIXED_C}, Last-Chunk Only)")
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, "sparsity_by_layer.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _plot_tradeoff(tradeoff_df: pd.DataFrame, output_dir: str) -> None:
    """Plot sparsity-accuracy tradeoff at the best layer."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    best_layer = tradeoff_df["layer"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_key in MODEL_KEYS:
        subset = tradeoff_df[tradeoff_df["model"] == model_key].sort_values("num_nonzero")
        ax.plot(subset["num_nonzero"], subset["roc_auc"],
                marker="o", markersize=5, label=MODEL_LABELS[model_key])

    ax.set_xlabel("Number of Nonzero Dimensions")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Sparsity-Accuracy Tradeoff (Layer {best_layer}, Last-Chunk Only)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(figures_dir, "sparsity_accuracy_tradeoff.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _plot_dimension_overlap(overlap_info: dict, output_dir: str) -> None:
    """Bar chart showing base-only, shared, distilled-only dimension counts."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    categories = ["Base only", "Shared", "Distilled only"]
    counts = [
        overlap_info["base_only_count"],
        overlap_info["shared_count"],
        overlap_info["distilled_only_count"],
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Number of Dimensions")
    ax.set_title(f"Nonzero Dimension Overlap at Layer {overlap_info['best_layer']} (C={overlap_info['C']})\n"
                 f"Jaccard = {overlap_info['jaccard_similarity']:.3f}, "
                 f"Weight r = {overlap_info['weight_correlation_shared_dims']:.3f}")
    ax.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(figures_dir, "dimension_overlap_venn.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _l1_stability_check(
    layer_data: dict, best_layer: int, overlap_info: dict, output_dir: str,
) -> None:
    """Test L1 feature selection stability via within-model Jaccard across splits."""
    metrics_dir = os.path.join(output_dir, "metrics")
    n_splits = 10

    X_base, X_dist, labels, is_last, pids = layer_data[best_layer]
    mask = is_last
    y = labels[mask]
    g = pids[mask]

    stability_results = {}
    for model_key, X_all in [("base", X_base), ("distilled", X_dist)]:
        X = X_all[mask]

        # Train on 10 different splits, collect nonzero sets
        nonzero_sets = []
        for seed in range(n_splits):
            gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
            train_idx, test_idx = next(gss.split(X, y, groups=g))

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])

            probe = LogisticRegression(
                solver="saga", C=FIXED_C, l1_ratio=1, class_weight="balanced",
                max_iter=5000, random_state=RANDOM_STATE,
            )
            probe.fit(X_tr, y[train_idx])
            nonzero = set(np.nonzero(probe.coef_[0])[0].tolist())
            nonzero_sets.append(nonzero)

        # Pairwise Jaccard for all 45 pairs
        pairwise = []
        for i, j in combinations(range(n_splits), 2):
            union = nonzero_sets[i] | nonzero_sets[j]
            inter = nonzero_sets[i] & nonzero_sets[j]
            jac = len(inter) / len(union) if union else 0.0
            pairwise.append(jac)

        stability_results[model_key] = {
            "pairwise_jaccards": pairwise,
            "mean": float(np.mean(pairwise)),
            "std": float(np.std(pairwise)),
            "min": float(np.min(pairwise)),
            "max": float(np.max(pairwise)),
            "n_splits": n_splits,
            "avg_nonzero_per_split": float(np.mean([len(s) for s in nonzero_sets])),
        }

    cross_model_jaccard = overlap_info["jaccard_similarity"]
    output = {
        "layer": best_layer,
        "C": FIXED_C,
        "cross_model_jaccard": cross_model_jaccard,
        "within_model": stability_results,
    }

    json_path = os.path.join(metrics_dir, "l1_stability_check.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved L1 stability check to {json_path}")

    # Print summary table
    print(f"\n  L1 Stability Check (layer {best_layer}, C={FIXED_C}):")
    print(f"  {'Model':<12} {'Mean Jaccard':>13} {'Std':>7} {'Min':>7} {'Max':>7} {'Avg dims':>9}")
    print(f"  {'-'*12} {'-'*13} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
    for model_key in MODEL_KEYS:
        s = stability_results[model_key]
        print(f"  {model_key:<12} {s['mean']:13.3f} {s['std']:7.3f} "
              f"{s['min']:7.3f} {s['max']:7.3f} {s['avg_nonzero_per_split']:9.1f}")
    print(f"  {'cross-model':<12} {cross_model_jaccard:13.3f}")
    print()

    # Interpretation
    avg_within = np.mean([stability_results[m]["mean"] for m in MODEL_KEYS])
    if avg_within < 0.1:
        print("  Interpretation: Within-model Jaccard is low — L1 feature selection")
        print("  is unstable. The low cross-model overlap may be an L1 artifact.")
    elif avg_within > 3 * cross_model_jaccard:
        print("  Interpretation: Within-model Jaccard is much higher than cross-model.")
        print("  L1 is stable within each model but selects different features across")
        print("  models. The cross-model difference is real.")
    else:
        print("  Interpretation: Within-model and cross-model Jaccard are comparable.")
        print("  Cannot distinguish real model differences from L1 instability.")


def _print_summary(
    results_df: pd.DataFrame, fixed_c_df: pd.DataFrame, overlap_info: dict,
) -> None:
    """Print summary of sparsity analysis."""
    print(f"\n{'='*60}")
    print("Sparsity Analysis Summary")
    print(f"{'='*60}")

    print("\n  Best-C grid search results:")
    for model_key in MODEL_KEYS:
        subset = results_df[results_df["model"] == model_key]
        best = subset.loc[subset["roc_auc"].idxmax()]
        print(f"    {model_key}: best layer {int(best['layer'])} "
              f"(AUC={best['roc_auc']:.4f}, C={best['best_C']}, "
              f"{int(best['num_nonzero'])} dims)")

    print(f"\n  Fixed C={FIXED_C} results:")
    for model_key in MODEL_KEYS:
        subset = fixed_c_df[fixed_c_df["model"] == model_key]
        best = subset.loc[subset["roc_auc"].idxmax()]
        avg_nonzero = subset["num_nonzero"].mean()
        print(f"    {model_key}: best layer {int(best['layer'])} "
              f"(AUC={best['roc_auc']:.4f}, {int(best['num_nonzero'])} dims), "
              f"avg {avg_nonzero:.0f} dims across layers")

    print(f"\n  Dimension Overlap (layer {overlap_info['best_layer']}, C={overlap_info['C']}):")
    print(f"    Base nonzero:      {overlap_info['base_nonzero_count']}")
    print(f"    Distilled nonzero: {overlap_info['distilled_nonzero_count']}")
    print(f"    Shared:            {overlap_info['shared_count']}")
    print(f"    Jaccard similarity: {overlap_info['jaccard_similarity']:.3f}")
    print(f"    Weight correlation (shared dims): {overlap_info['weight_correlation_shared_dims']:.3f}")


def sparsity_analysis(
    hidden_states_dir: str, chunks_path: str,
    probe_results_path: str, output_dir: str,
) -> None:
    """Run full L1 sparsity analysis."""
    print(f"Loading chunks from {chunks_path}...")
    problems = load_jsonl(chunks_path)
    print(f"Loaded {len(problems)} problems")

    print("\nLoading aligned hidden states...")
    layer_data = _load_aligned_data(hidden_states_dir, problems)
    print(f"Loaded {len(layer_data)} layers")

    # Find best layer from Day 6 probe results
    print("\nSelecting best layer from probe results...")
    best_layer = _get_best_layer(probe_results_path)

    # Core: L1 grid search across all layers
    print("\n--- L1 Sparsity Analysis (best-C grid search, last-chunk only) ---")
    results_df = _core_sparsity(layer_data, output_dir)

    # Fixed-C sweep for smooth curves
    print(f"\n--- Fixed C={FIXED_C} Sparsity (last-chunk only) ---")
    fixed_c_df = _fixed_c_sparsity(layer_data, output_dir)

    # Additional analyses at best layer
    print(f"\n--- Sparsity-Accuracy Tradeoff (layer {best_layer}) ---")
    tradeoff_df = _sparsity_tradeoff(layer_data, best_layer, output_dir)

    print(f"\n--- Dimension Overlap (layer {best_layer}, C={FIXED_C}) ---")
    overlap_info = _dimension_overlap(layer_data, best_layer, output_dir)

    # Figures
    _plot_sparsity_by_layer(fixed_c_df, output_dir)
    _plot_tradeoff(tradeoff_df, output_dir)
    _plot_dimension_overlap(overlap_info, output_dir)

    _print_summary(results_df, fixed_c_df, overlap_info)

    # L1 stability check
    print(f"\n--- L1 Stability Check (layer {best_layer}, C={FIXED_C}) ---")
    _l1_stability_check(layer_data, best_layer, overlap_info, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1 sparsity analysis with dimension overlap",
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

    sparsity_analysis(
        hidden_states_dir=args.hidden_states_dir,
        chunks_path=args.chunks_path,
        probe_results_path=args.probe_results_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
