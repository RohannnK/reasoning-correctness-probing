"""CKA representational similarity analysis and permutation test for probe gap significance."""

import argparse
import os

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

TEST_SIZE = 0.2
RANDOM_STATE = 42
PERM_LAYERS = [0, 7, 14, 21, 28]
N_PERMUTATIONS = 1000


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two representation matrices.

    Args:
        X: Representation matrix of shape (n_samples, d1).
        Y: Representation matrix of shape (n_samples, d2).

    Returns:
        Linear CKA similarity value in [0, 1].
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0

    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def _load_aligned_data(
    hidden_states_dir: str, problems: list[dict],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load aligned hidden states for both base and distilled models.

    Returns:
        Dict mapping layer_idx to (X_base, X_dist, labels, is_last, problem_ids).
        Row i in X_base corresponds to the same problem/chunk as row i in X_dist.
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


def _compute_layer_cka(
    layer_data: dict, output_dir: str,
) -> pd.DataFrame:
    """Compute layer-matched CKA between base and distilled."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    results = []
    for layer_idx in sorted(layer_data.keys()):
        X_base, X_dist, _, _, _ = layer_data[layer_idx]
        cka_val = linear_cka(X_base, X_dist)
        results.append({
            "layer": layer_idx,
            "cka": cka_val,
            "cka_divergence": 1.0 - cka_val,
        })
        print(f"  Layer {layer_idx:2d}: CKA = {cka_val:.4f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(metrics_dir, "cka_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved layer-matched CKA to {csv_path}")
    return df


def _compute_cross_layer_cka(
    layer_data: dict, output_dir: str,
) -> pd.DataFrame:
    """Compute cross-layer CKA: base layer i vs distilled layer j."""
    metrics_dir = os.path.join(output_dir, "metrics")
    layers = sorted(layer_data.keys())
    n = len(layers)

    # Collect all base and distilled matrices
    base_mats = {l: layer_data[l][0] for l in layers}
    dist_mats = {l: layer_data[l][1] for l in layers}

    rows = []
    matrix = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            cka_val = linear_cka(base_mats[li], dist_mats[lj])
            matrix[i, j] = cka_val
            rows.append({"base_layer": li, "distilled_layer": lj, "cka": cka_val})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(metrics_dir, "cka_cross_layer.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved cross-layer CKA to {csv_path}")
    return df


def _compute_within_model_cka(layer_data: dict, output_dir: str) -> None:
    """Compute within-model cross-layer CKA for base and distilled separately."""
    metrics_dir = os.path.join(output_dir, "metrics")
    layers = sorted(layer_data.keys())

    for model_idx, model_name in [(0, "base"), (1, "distilled")]:
        mats = {l: layer_data[l][model_idx] for l in layers}
        rows = []
        for li in layers:
            for lj in layers:
                cka_val = linear_cka(mats[li], mats[lj])
                rows.append({"layer_i": li, "layer_j": lj, "cka": cka_val})

        df = pd.DataFrame(rows)
        csv_path = os.path.join(metrics_dir, f"cka_within_{model_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved within-{model_name} CKA to {csv_path}")


def _plot_cka_by_layer(cka_df: pd.DataFrame, output_dir: str) -> None:
    """Line plot of CKA similarity vs layer."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cka_df["layer"], cka_df["cka"], marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("Representational Similarity: Base vs Distilled")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(figures_dir, "cka_by_layer.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _plot_cross_layer_heatmap(cross_df: pd.DataFrame, output_dir: str) -> None:
    """29x29 heatmap of cross-layer CKA."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    layers = sorted(cross_df["base_layer"].unique())
    n = len(layers)
    matrix = cross_df.pivot(
        index="distilled_layer", columns="base_layer", values="cka",
    ).values

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, origin="lower")
    fig.colorbar(im, ax=ax, label="CKA")

    # Diagonal reference line
    ax.plot([0, n - 1], [0, n - 1], "r--", linewidth=1, alpha=0.7)

    tick_step = max(1, n // 10)
    tick_positions = list(range(0, n, tick_step))
    tick_labels = [str(layers[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Base Layer")
    ax.set_ylabel("Distilled Layer")
    ax.set_title("Cross-Layer CKA: Base vs Distilled")

    save_path = os.path.join(figures_dir, "cka_cross_layer_heatmap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def _plot_cka_vs_probe_gap(
    cka_df: pd.DataFrame, probe_results_path: str, output_dir: str,
) -> None:
    """Scatter plot of CKA divergence vs probe accuracy gap."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    probe_df = pd.read_csv(probe_results_path)
    last_linear = probe_df[
        (probe_df["analysis_type"] == "last_chunk_only")
        & (probe_df["probe_type"] == "linear")
    ]

    # Compute per-layer probe gap: distilled - base
    base_auc = last_linear[last_linear["model_key"] == "base"].set_index("layer")["roc_auc"]
    dist_auc = last_linear[last_linear["model_key"] == "distilled"].set_index("layer")["roc_auc"]
    common_layers = sorted(set(base_auc.index) & set(dist_auc.index) & set(cka_df["layer"]))

    probe_gap = dist_auc.loc[common_layers].values - base_auc.loc[common_layers].values
    cka_div = cka_df.set_index("layer").loc[common_layers, "cka_divergence"].values

    r, p = pearsonr(cka_div, probe_gap)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(cka_div, probe_gap, c=common_layers, cmap="viridis", s=40, edgecolors="k", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="Layer")

    ax.set_xlabel("CKA Divergence (1 - CKA)")
    ax.set_ylabel("Probe AUC Gap (Distilled - Base)")
    ax.set_title("CKA Divergence vs Probe Accuracy Gap")
    ax.annotate(f"r = {r:.3f}, p = {p:.4f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=10, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(figures_dir, "cka_vs_probe_gap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path} (Pearson r={r:.3f}, p={p:.4f})")


def _train_probe_auc(X_train, y_train, X_test, y_test) -> float:
    """Train a simple logistic regression probe and return test ROC-AUC."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    lr = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000, solver="lbfgs",
    )
    lr.fit(X_tr, y_train)
    probs = lr.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_test, probs)


def _permutation_test(
    layer_data: dict, problems: list[dict], output_dir: str,
) -> pd.DataFrame:
    """Permutation test for probe gap significance on selected layers."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    rng = np.random.RandomState(RANDOM_STATE)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    results = []
    for layer_idx in PERM_LAYERS:
        if layer_idx not in layer_data:
            continue

        X_base, X_dist, labels, is_last, pids = layer_data[layer_idx]

        # Filter to last chunks only
        mask = is_last
        X_b = X_base[mask]
        X_d = X_dist[mask]
        y = labels[mask]
        g = pids[mask]

        if len(np.unique(y)) < 2:
            print(f"  Layer {layer_idx}: skipping (single class in last chunks)")
            continue

        # Single train/test split for consistency
        train_idx, test_idx = next(gss.split(X_b, y, groups=g))

        # Observed gap
        auc_base = _train_probe_auc(X_b[train_idx], y[train_idx], X_b[test_idx], y[test_idx])
        auc_dist = _train_probe_auc(X_d[train_idx], y[train_idx], X_d[test_idx], y[test_idx])
        observed_gap = auc_dist - auc_base

        # Permutation: shuffle model labels per problem
        unique_pids = np.unique(g)
        perm_gaps = []
        for perm_i in range(N_PERMUTATIONS):
            # For each problem, randomly assign to "base" or "distilled"
            swap_mask = rng.randint(0, 2, size=len(unique_pids)).astype(bool)
            swap_pids = set(unique_pids[swap_mask])

            X_perm_b = X_b.copy()
            X_perm_d = X_d.copy()
            for idx in range(len(g)):
                if g[idx] in swap_pids:
                    X_perm_b[idx], X_perm_d[idx] = X_d[idx].copy(), X_b[idx].copy()

            try:
                auc_pb = _train_probe_auc(
                    X_perm_b[train_idx], y[train_idx], X_perm_b[test_idx], y[test_idx],
                )
                auc_pd = _train_probe_auc(
                    X_perm_d[train_idx], y[train_idx], X_perm_d[test_idx], y[test_idx],
                )
                perm_gaps.append(auc_pd - auc_pb)
            except ValueError:
                continue

        p_value = np.mean([g >= observed_gap for g in perm_gaps]) if perm_gaps else np.nan
        results.append({
            "layer": layer_idx,
            "observed_gap": observed_gap,
            "p_value": p_value,
            "significant_at_05": p_value < 0.05 if not np.isnan(p_value) else False,
        })
        print(f"  Layer {layer_idx:2d}: gap={observed_gap:+.4f}, "
              f"p={p_value:.4f}, sig={'*' if p_value < 0.05 else ''}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(metrics_dir, "permutation_test.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved permutation test results to {csv_path}")
    return df


def _print_summary(cka_df: pd.DataFrame, perm_df: pd.DataFrame) -> None:
    """Print summary of CKA and permutation test findings."""
    print(f"\n{'='*60}")
    print("CKA Analysis Summary")
    print(f"{'='*60}")

    print(f"\n  Layer-matched CKA (base vs distilled):")
    print(f"    Mean CKA:  {cka_df['cka'].mean():.4f}")
    print(f"    Min CKA:   {cka_df['cka'].min():.4f} (layer {cka_df.loc[cka_df['cka'].idxmin(), 'layer']:.0f})")
    print(f"    Max CKA:   {cka_df['cka'].max():.4f} (layer {cka_df.loc[cka_df['cka'].idxmax(), 'layer']:.0f})")

    if not perm_df.empty:
        print(f"\n  Permutation Test (H0: distilled AUC = base AUC, last-chunk-only):")
        print(f"  {'Layer':>6}  {'Obs. Gap':>10}  {'p-value':>8}  {'Sig.':>5}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*5}")
        for _, row in perm_df.iterrows():
            sig = "*" if row["significant_at_05"] else ""
            print(f"  {int(row['layer']):6d}  {row['observed_gap']:+10.4f}  "
                  f"{row['p_value']:8.4f}  {sig:>5}")


def cka_analysis(
    hidden_states_dir: str, chunks_path: str,
    probe_results_path: str, output_dir: str,
) -> None:
    """Run full CKA analysis and permutation test."""
    print(f"Loading chunks from {chunks_path}...")
    problems = load_jsonl(chunks_path)
    print(f"Loaded {len(problems)} problems")

    print("\nLoading aligned hidden states...")
    layer_data = _load_aligned_data(hidden_states_dir, problems)
    print(f"Loaded {len(layer_data)} layers")

    # Part 1: CKA analysis
    print("\n--- Layer-matched CKA ---")
    cka_df = _compute_layer_cka(layer_data, output_dir)

    print("\n--- Cross-layer CKA (base x distilled) ---")
    cross_df = _compute_cross_layer_cka(layer_data, output_dir)

    print("\n--- Within-model cross-layer CKA ---")
    _compute_within_model_cka(layer_data, output_dir)

    _plot_cka_by_layer(cka_df, output_dir)
    _plot_cross_layer_heatmap(cross_df, output_dir)
    _plot_cka_vs_probe_gap(cka_df, probe_results_path, output_dir)

    # Part 2: Permutation test
    print(f"\n--- Permutation test (layers {PERM_LAYERS}, {N_PERMUTATIONS} permutations) ---")
    perm_df = _permutation_test(layer_data, problems, output_dir)

    _print_summary(cka_df, perm_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CKA representational similarity and permutation test",
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

    cka_analysis(
        hidden_states_dir=args.hidden_states_dir,
        chunks_path=args.chunks_path,
        probe_results_path=args.probe_results_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
