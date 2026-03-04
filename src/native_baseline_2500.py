"""Native baseline (2500 problems): base model on its own CoT vs distilled-text CoT.

Full replication of the 500-problem native baseline experiment. Trains
last-chunk-only linear probes at every layer on base-model hidden states from
the base model's *own* native CoT, then compares AUCs against the main
experiment (base model processing distilled-model CoT text).

Usage:
    python3 -m src.native_baseline_2500
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.train_probes import (
    RANDOM_STATE,
    TEST_SIZE,
    _load_layer_data,
    _train_linear,
)
from src.utils import load_jsonl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NATIVE_CHUNKS_PATH = "data/chunks/gsm8k_base_native_chunks_2500.jsonl"
NATIVE_HS_DIR = "data/hidden_states/base_native_2500"
MAIN_RESULTS_CSV = "results/metrics/probe_results.csv"
OUTPUT_CSV = "results/metrics/native_baseline_2500_results.csv"

N_SPLITS = 10


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def run_native_probes() -> list[dict]:
    """Train last-chunk-only linear probes at every layer on native CoT data."""
    problems = load_jsonl(NATIVE_CHUNKS_PATH)
    print(f"Loaded {len(problems)} native problems")

    layer_data = _load_layer_data(NATIVE_HS_DIR, problems)
    if not layer_data:
        raise RuntimeError(f"No hidden states found in {NATIVE_HS_DIR}")
    print(f"Loaded {len(layer_data)} layers")

    rows = []
    for layer_idx, (X, y, _positions, is_last, problem_ids) in sorted(layer_data.items()):
        X_last = X[is_last]
        y_last = y[is_last]
        pids_last = problem_ids[is_last]

        if len(np.unique(y_last)) < 2:
            print(f"  Layer {layer_idx:2d}: only one class present, skipping")
            continue

        gss = GroupShuffleSplit(
            n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        split_aucs: list[float] = []
        for train_idx, test_idx in gss.split(X_last, y_last, groups=pids_last):
            X_tr, X_te = X_last[train_idx], X_last[test_idx]
            y_tr, y_te = y_last[train_idx], y_last[test_idx]
            if len(np.unique(y_te)) < 2:
                continue
            metrics, _ = _train_linear(X_tr, y_tr, X_te, y_te)
            split_aucs.append(metrics["roc_auc"])

        if not split_aucs:
            print(f"  Layer {layer_idx:2d}: all splits single-class, skipping")
            continue

        mean_auc = float(np.mean(split_aucs))
        std_auc = float(np.std(split_aucs, ddof=1)) if len(split_aucs) > 1 else 0.0
        ci_lo = mean_auc - 1.96 * std_auc
        ci_hi = mean_auc + 1.96 * std_auc

        rows.append({
            "condition": "base_native_2500",
            "layer": layer_idx,
            "roc_auc": mean_auc,
            "roc_auc_ci_lo": ci_lo,
            "roc_auc_ci_hi": ci_hi,
            "n_samples": int(is_last.sum()),
            "pct_correct": float(y_last.mean()),
        })
        print(f"  Layer {layer_idx:2d}: AUC={mean_auc:.4f} ± {std_auc:.4f}"
              f"  n={int(is_last.sum())}  pct_correct={y_last.mean():.3f}"
              f"  (valid splits={len(split_aucs)}/{N_SPLITS})")

    return rows


# ---------------------------------------------------------------------------
# Comparison with main experiment
# ---------------------------------------------------------------------------

def load_main_base_results() -> pd.DataFrame:
    """Load base model last-chunk-only linear results from the main experiment."""
    df = pd.read_csv(MAIN_RESULTS_CSV)
    base_last = df[
        (df["model_key"] == "base")
        & (df["probe_type"] == "linear")
        & (df["analysis_type"] == "last_chunk_only")
    ][["layer", "roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi"]].copy()
    base_last["condition"] = "base_on_distilled_text"
    return base_last


def print_comparison(native_df: pd.DataFrame, main_df: pd.DataFrame) -> None:
    native = native_df.set_index("layer")
    main = main_df.set_index("layer")

    print("\n" + "=" * 90)
    print(f"{'COMPARISON: Base-native (2500) vs Base-on-distilled-text (last-chunk-only linear)':^90}")
    print("=" * 90)
    header = (f"{'Layer':>5}  {'Base-on-distilled':>22}  {'Base-native':>22}  {'Gap':>10}")
    print(header)
    print("-" * 90)

    all_layers = sorted(set(native.index) | set(main.index))
    for layer in all_layers:
        m_auc = main.loc[layer, "roc_auc"] if layer in main.index else float("nan")
        m_lo = main.loc[layer, "roc_auc_ci_lo"] if layer in main.index else float("nan")
        m_hi = main.loc[layer, "roc_auc_ci_hi"] if layer in main.index else float("nan")
        n_auc = native.loc[layer, "roc_auc"] if layer in native.index else float("nan")
        n_lo = native.loc[layer, "roc_auc_ci_lo"] if layer in native.index else float("nan")
        n_hi = native.loc[layer, "roc_auc_ci_hi"] if layer in native.index else float("nan")
        gap = n_auc - m_auc

        marker = ""
        if layer == 17:
            marker = "  ← main best"

        print(f"{layer:>5}  {m_auc:.4f} [{m_lo:.2f},{m_hi:.2f}]"
              f"  {n_auc:.4f} [{n_lo:.2f},{n_hi:.2f}]"
              f"  {gap:>+10.4f}{marker}")

    print("-" * 90)

    # Best layers
    if not main.empty:
        m_best_layer = int(main["roc_auc"].idxmax())
        m_best_auc = main.loc[m_best_layer, "roc_auc"]
        print(f"  Best main (base-on-distilled): layer {m_best_layer:2d}  AUC={m_best_auc:.4f}")
    if not native.empty:
        n_best_layer = int(native["roc_auc"].idxmax())
        n_best_auc = native.loc[n_best_layer, "roc_auc"]
        print(f"  Best native (base-native):     layer {n_best_layer:2d}  AUC={n_best_auc:.4f}")

    # CI overlap check at best main layer
    if not main.empty and not native.empty and m_best_layer in native.index:
        m_lo = main.loc[m_best_layer, "roc_auc_ci_lo"]
        m_hi = main.loc[m_best_layer, "roc_auc_ci_hi"]
        n_lo = native.loc[m_best_layer, "roc_auc_ci_lo"]
        n_hi = native.loc[m_best_layer, "roc_auc_ci_hi"]
        overlap = m_lo <= n_hi and n_lo <= m_hi
        gap = native.loc[m_best_layer, "roc_auc"] - main.loc[m_best_layer, "roc_auc"]
        print(f"\n  At main best layer {m_best_layer}: "
              f"main={m_best_auc:.3f}, native={native.loc[m_best_layer, 'roc_auc']:.3f}, "
              f"gap={gap:+.3f}, CIs overlap? {'Y' if overlap else 'N'}")


def print_interpretation(native_df: pd.DataFrame, main_df: pd.DataFrame) -> None:
    native = native_df.set_index("layer")
    main = main_df.set_index("layer")
    common = native.index.intersection(main.index)

    if common.empty:
        return

    gap_values = [native.loc[l, "roc_auc"] - main.loc[l, "roc_auc"] for l in common]
    mean_gap = float(np.mean(gap_values))

    # Get main best AUC for interpretation thresholds
    main_best = main["roc_auc"].max()
    native_best = native["roc_auc"].max() if not native.empty else float("nan")

    print("\n" + "=" * 90)
    print("INTERPRETIVE SUMMARY")
    print("=" * 90)
    print(f"  Mean AUC gap across all layers (native − distilled-text): {mean_gap:+.4f}")
    print(f"  Main best AUC (base-on-distilled L17): {main_best:.4f}")
    print(f"  Native best AUC: {native_best:.4f}")

    # Class balance note
    if not native.empty:
        pct = native_df.iloc[0].get("pct_correct", float("nan"))
        print(f"\n  NOTE: Class balance differs:")
        print(f"    - Main experiment (distilled text): ~19.6% correct (heavily imbalanced)")
        print(f"    - Native baseline: ~{pct*100:.1f}% correct")
        print(f"    - ROC-AUC is threshold-invariant, so class balance doesn't directly bias it")
        print(f"    - Native chains are typically shorter (fewer chunks per problem)")

    print()
    if abs(mean_gap) < 0.03:
        print("  INTERPRETATION: |gap| < 0.03 → OOD confound is MINIMAL.")
        print("  The base model's correctness signal is robust to input format.")
        print("  This strengthens the main finding: the AUC gap between base and")
        print("  distilled reflects representational geometry, not <think>-format OOD.")
    elif mean_gap > 0.03:
        print("  INTERPRETATION: native >> distilled-text → <think> format was SUPPRESSING")
        print("  the base model's correctness signal.")
        print("  The base model has stronger native correctness encoding; the OOD format")
        print("  impairs probing. Main results may UNDERESTIMATE base representational capacity.")
    else:
        print("  INTERPRETATION: native << distilled-text → <think> format was HELPING.")
        print("  The structured reasoning format in distilled text aids base model probing.")
        print("  Format effects (not geometry alone) partially drive the base probe AUC.")
        print("  The gap between base and distilled may be partly a format artifact.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 90)
    print("Native Baseline Analysis — 2500 Problems")
    print("=" * 90)

    # Step 1: Run probes on native hidden states
    print("\n--- Training probes on base-native hidden states ---")
    native_rows = run_native_probes()
    native_df = pd.DataFrame(native_rows)

    # Step 2: Load main experiment results (no re-running)
    print("\n--- Loading main experiment results ---")
    main_df = load_main_base_results()
    print(f"  Loaded {len(main_df)} layers from {MAIN_RESULTS_CSV}")

    # Step 3: Build combined CSV
    main_rows = []
    for _, row in main_df.iterrows():
        main_rows.append({
            "condition": "base_on_distilled_text",
            "layer": int(row["layer"]),
            "roc_auc": row["roc_auc"],
            "roc_auc_ci_lo": row["roc_auc_ci_lo"],
            "roc_auc_ci_hi": row["roc_auc_ci_hi"],
            "n_samples": 2500,  # main experiment had 2500 problems
            "pct_correct": 0.196,  # ~19.6% correct in main
        })
    combined_df = pd.concat([
        pd.DataFrame(main_rows),
        native_df,
    ], ignore_index=True)
    combined_df = combined_df[
        ["condition", "layer", "roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi",
         "n_samples", "pct_correct"]
    ]

    os.makedirs("results/metrics", exist_ok=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved combined results to {OUTPUT_CSV}")

    # Step 4: Print comparison table
    print_comparison(native_df, main_df)

    # Step 5: Interpretation
    print_interpretation(native_df, main_df)


if __name__ == "__main__":
    main()
