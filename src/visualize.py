"""Generate paper-ready figures for the probing analysis."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_style() -> None:
    """Configure matplotlib for paper-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    sns.set_palette("colorblind")


def plot_layer_auc(
    probe_results_paths: dict[str, str],
    output_path: str,
) -> None:
    """Plot layer-wise ROC-AUC for base vs distilled models.

    Args:
        probe_results_paths: Mapping from model label to probe results CSV path.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, path in probe_results_paths.items():
        df = pd.read_csv(path)
        lr_df = df[df["probe_type"] == "logistic_regression"]
        ax.plot(lr_df["layer"], lr_df["roc_auc"], marker="o", markersize=4, label=f"{label} (LR)")

        mlp_df = df[df["probe_type"] == "mlp"]
        ax.plot(mlp_df["layer"], mlp_df["roc_auc"], marker="s", markersize=4,
                linestyle="--", label=f"{label} (MLP)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Probe Accuracy by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved layer AUC plot to {output_path}")


def plot_cka(cka_path: str, output_path: str) -> None:
    """Plot layer-wise CKA between base and distilled models.

    Args:
        cka_path: Path to CKA results CSV.
        output_path: Path to save the figure.
    """
    df = pd.read_csv(cka_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["layer"], df["cka"], marker="o", markersize=5, color="teal")
    ax.fill_between(df["layer"], df["cka"], alpha=0.15, color="teal")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Representational Similarity (Base vs. Distilled)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved CKA plot to {output_path}")


def plot_cka_vs_accuracy_gap(
    cka_path: str,
    base_probe_path: str,
    distilled_probe_path: str,
    output_path: str,
) -> None:
    """Scatter plot of CKA divergence vs probe accuracy gap.

    Args:
        cka_path: Path to CKA results CSV.
        base_probe_path: Path to base model probe results CSV.
        distilled_probe_path: Path to distilled model probe results CSV.
        output_path: Path to save the figure.
    """
    cka_df = pd.read_csv(cka_path)
    base_df = pd.read_csv(base_probe_path)
    dist_df = pd.read_csv(distilled_probe_path)

    # Use LR probes for accuracy gap
    base_lr = base_df[base_df["probe_type"] == "logistic_regression"].set_index("layer")
    dist_lr = dist_df[dist_df["probe_type"] == "logistic_regression"].set_index("layer")
    cka_indexed = cka_df.set_index("layer")

    common = sorted(set(base_lr.index) & set(dist_lr.index) & set(cka_indexed.index))

    cka_divergence = [1.0 - cka_indexed.loc[l, "cka"] for l in common]
    auc_gap = [dist_lr.loc[l, "roc_auc"] - base_lr.loc[l, "roc_auc"] for l in common]

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(cka_divergence, auc_gap, c=common, cmap="viridis",
                         s=50, edgecolors="black", linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Layer Index")

    ax.set_xlabel("CKA Divergence (1 - CKA)")
    ax.set_ylabel("AUC Gap (Distilled - Base)")
    ax.set_title("Representational Divergence vs. Probe Accuracy Gap")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved CKA vs accuracy gap plot to {output_path}")


def plot_sparsity(
    sparsity_paths: dict[str, str],
    output_path: str,
) -> None:
    """Plot sparsity (number of nonzero coefficients) by layer.

    Args:
        sparsity_paths: Mapping from model label to sparsity results CSV.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, path in sparsity_paths.items():
        df = pd.read_csv(path)
        # Use best C (highest accuracy) per layer
        best = df.loc[df.groupby("layer")["test_accuracy"].idxmax()]
        ax.plot(best["layer"], best["n_nonzero"], marker="o", markersize=4, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Nonzero Coefficients")
    ax.set_title("Sparsity of Correctness Representations by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved sparsity plot to {output_path}")


def plot_early_exit(
    early_exit_paths: dict[str, str],
    output_path: str,
) -> None:
    """Plot Pareto frontier of early exit: accuracy vs layers saved.

    Args:
        early_exit_paths: Mapping from model label to early exit CSV.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, path in early_exit_paths.items():
        df = pd.read_csv(path)
        ax.plot(df["layers_saved_pct"], df["accuracy"], marker="o", markersize=6, label=label)
        for _, row in df.iterrows():
            ax.annotate(f'{row["threshold"]:.2f}',
                        (row["layers_saved_pct"], row["accuracy"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Layers Saved (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Early Exit Pareto Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved early exit plot to {output_path}")


def generate_all_figures(
    results_dir: str,
    figures_dir: str,
    base_short_name: str = "qwen-base",
    distilled_short_name: str = "r1-distill",
) -> None:
    """Generate all paper figures from results directory.

    Args:
        results_dir: Directory containing all result CSVs.
        figures_dir: Directory to save figures.
        base_short_name: Short name of the base model.
        distilled_short_name: Short name of the distilled model.
    """
    os.makedirs(figures_dir, exist_ok=True)
    setup_style()

    probes_dir = os.path.join(results_dir, "probes")
    base_probe = os.path.join(probes_dir, f"{base_short_name}_probe_results.csv")
    dist_probe = os.path.join(probes_dir, f"{distilled_short_name}_probe_results.csv")

    # 1. Layer-wise ROC-AUC
    if os.path.exists(base_probe) and os.path.exists(dist_probe):
        plot_layer_auc(
            {base_short_name: base_probe, distilled_short_name: dist_probe},
            os.path.join(figures_dir, "layer_auc.pdf"),
        )

    # 2. Layer-wise CKA
    cka_path = os.path.join(results_dir, "cka_results.csv")
    if os.path.exists(cka_path):
        plot_cka(cka_path, os.path.join(figures_dir, "cka_similarity.pdf"))

    # 3. CKA divergence vs probe accuracy gap
    if os.path.exists(cka_path) and os.path.exists(base_probe) and os.path.exists(dist_probe):
        plot_cka_vs_accuracy_gap(
            cka_path, base_probe, dist_probe,
            os.path.join(figures_dir, "cka_vs_accuracy_gap.pdf"),
        )

    # 4. Sparsity by layer
    base_sparsity = os.path.join(results_dir, f"{base_short_name}_sparsity.csv")
    dist_sparsity = os.path.join(results_dir, f"{distilled_short_name}_sparsity.csv")
    if os.path.exists(base_sparsity) and os.path.exists(dist_sparsity):
        plot_sparsity(
            {base_short_name: base_sparsity, distilled_short_name: dist_sparsity},
            os.path.join(figures_dir, "sparsity_by_layer.pdf"),
        )

    # 5. Early exit Pareto frontier
    base_exit = os.path.join(results_dir, f"{base_short_name}_early_exit.csv")
    dist_exit = os.path.join(results_dir, f"{distilled_short_name}_early_exit.csv")
    if os.path.exists(base_exit) and os.path.exists(dist_exit):
        plot_early_exit(
            {base_short_name: base_exit, distilled_short_name: dist_exit},
            os.path.join(figures_dir, "early_exit_pareto.pdf"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--figures_dir", type=str, required=True)
    parser.add_argument("--base_short_name", type=str, default="qwen-base")
    parser.add_argument("--distilled_short_name", type=str, default="r1-distill")
    args = parser.parse_args()

    generate_all_figures(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        base_short_name=args.base_short_name,
        distilled_short_name=args.distilled_short_name,
    )


if __name__ == "__main__":
    main()
