"""Analyze native baseline: base model on its own CoT vs base model on distilled-text CoT.

Compares two conditions at every layer (last-chunk-only, linear probe,
GroupShuffleSplit × 10 for CI):
  - base_native:            base model hidden states from its own native CoT
  - base_on_distilled_text: base model hidden states from distilled-model <think> text

Moves/unzips new data files into place, runs probes, prints side-by-side
comparison + interpretive summary, and saves results/metrics/native_baseline_results.csv.
"""

import os
import shutil
import zipfile

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
_DOWNLOADS = os.path.expanduser("~/Downloads")

_NATIVE_COT_SRC = os.path.join(_DOWNLOADS, "gsm8k_base_native_cot.jsonl")
_NATIVE_CHUNKS_SRC = os.path.join(_DOWNLOADS, "gsm8k_base_native_chunks.jsonl")
_NATIVE_HS_ZIP = os.path.join(_DOWNLOADS, "base_native_hidden_states.zip")

_NATIVE_COT_DST = "data/cot_outputs/gsm8k_base_native_cot.jsonl"
_NATIVE_CHUNKS_DST = "data/chunks/gsm8k_base_native_chunks.jsonl"
_NATIVE_HS_DIR = "data/hidden_states/base_native"

_EXISTING_CHUNKS = "data/chunks/gsm8k_chunks.jsonl"
_EXISTING_HS_DIR = "data/hidden_states/base"

OUTPUT_CSV = "results/metrics/native_baseline_results.csv"

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
N_SPLITS = 10
KEY_LAYERS = [0, 14, 24, 25, 28]

CONDITIONS: dict[str, tuple[str, str]] = {
    "base_native": (_NATIVE_HS_DIR, _NATIVE_CHUNKS_DST),
    "base_on_distilled_text": (_EXISTING_HS_DIR, _EXISTING_CHUNKS),
}


# ---------------------------------------------------------------------------
# File setup
# ---------------------------------------------------------------------------

def setup_files() -> None:
    """Copy/unzip new data files into the repo (idempotent)."""
    # CoT outputs
    os.makedirs("data/cot_outputs", exist_ok=True)
    if not os.path.exists(_NATIVE_COT_DST):
        shutil.copy(_NATIVE_COT_SRC, _NATIVE_COT_DST)
        print(f"Copied {_NATIVE_COT_SRC} → {_NATIVE_COT_DST}")
    else:
        print(f"Already present: {_NATIVE_COT_DST}")

    # Chunks
    os.makedirs("data/chunks", exist_ok=True)
    if not os.path.exists(_NATIVE_CHUNKS_DST):
        shutil.copy(_NATIVE_CHUNKS_SRC, _NATIVE_CHUNKS_DST)
        print(f"Copied {_NATIVE_CHUNKS_SRC} → {_NATIVE_CHUNKS_DST}")
    else:
        print(f"Already present: {_NATIVE_CHUNKS_DST}")

    # Hidden states zip → base_native/
    os.makedirs(_NATIVE_HS_DIR, exist_ok=True)
    existing_npz = [f for f in os.listdir(_NATIVE_HS_DIR) if f.endswith(".npz")]
    if existing_npz:
        print(f"Already present: {_NATIVE_HS_DIR}/ ({len(existing_npz)} .npz files)")
    else:
        print(f"Unzipping {_NATIVE_HS_ZIP} → {_NATIVE_HS_DIR}/")
        with zipfile.ZipFile(_NATIVE_HS_ZIP) as zf:
            members = [m for m in zf.namelist() if m.endswith(".npz")]
            for member in members:
                fname = os.path.basename(member)
                target = os.path.join(_NATIVE_HS_DIR, fname)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        extracted = len([f for f in os.listdir(_NATIVE_HS_DIR) if f.endswith(".npz")])
        print(f"Extracted {extracted} .npz files to {_NATIVE_HS_DIR}/")


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def run_last_chunk_probes(hs_dir: str, chunks_path: str) -> list[dict]:
    """Train last-chunk-only linear probes at every layer for one condition.

    Uses GroupShuffleSplit with N_SPLITS iterations; reports mean AUC and
    95% CI from mean ± 1.96 * std across splits.
    """
    problems = load_jsonl(chunks_path)
    layer_data = _load_layer_data(hs_dir, problems)
    if not layer_data:
        raise RuntimeError(f"No hidden states found in {hs_dir}")

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
# Output
# ---------------------------------------------------------------------------

def print_comparison(results_by_condition: dict[str, pd.DataFrame]) -> None:
    native = results_by_condition["base_native"].set_index("layer")
    dist = results_by_condition["base_on_distilled_text"].set_index("layer")

    print("\n" + "=" * 72)
    print(f"{'SIDE-BY-SIDE COMPARISON (last-chunk-only linear probe)':^72}")
    print("=" * 72)
    header = f"{'Layer':>6}  {'base_native':>14}  {'base_on_dist':>14}  {'gap (nat-dist)':>14}"
    print(header)
    print("-" * 72)
    for layer in KEY_LAYERS:
        n_auc = native.loc[layer, "roc_auc"] if layer in native.index else float("nan")
        d_auc = dist.loc[layer, "roc_auc"] if layer in dist.index else float("nan")
        gap = n_auc - d_auc
        print(f"{layer:>6}  {n_auc:>14.4f}  {d_auc:>14.4f}  {gap:>+14.4f}")
    print("-" * 72)

    if not native.empty:
        bn_best_layer = int(native["roc_auc"].idxmax())
        bn_best_auc = native.loc[bn_best_layer, "roc_auc"]
        print(f"  Best base_native:            layer {bn_best_layer:2d}  AUC={bn_best_auc:.4f}")
    if not dist.empty:
        bd_best_layer = int(dist["roc_auc"].idxmax())
        bd_best_auc = dist.loc[bd_best_layer, "roc_auc"]
        print(f"  Best base_on_distilled_text: layer {bd_best_layer:2d}  AUC={bd_best_auc:.4f}")


def print_interpretation(gap_values: list[float]) -> None:
    if not gap_values:
        return
    mean_gap = float(np.mean(gap_values))

    print("\n" + "=" * 72)
    print("INTERPRETIVE SUMMARY")
    print("=" * 72)
    print(f"  Mean AUC gap across all layers (native − distilled-text): {mean_gap:+.4f}")
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
    print("=" * 72)
    print("Native Baseline Analysis")
    print("=" * 72)

    # Step 1: Move files into place
    print("\n--- File setup ---")
    setup_files()

    # Step 2: Run probes for both conditions
    all_rows: list[dict] = []
    results_by_condition: dict[str, pd.DataFrame] = {}

    for condition, (hs_dir, chunks_path) in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"  hidden states: {hs_dir}")
        print(f"  chunks:        {chunks_path}")
        rows = run_last_chunk_probes(hs_dir, chunks_path)
        for r in rows:
            r["condition"] = condition
        all_rows.extend(rows)
        results_by_condition[condition] = pd.DataFrame(rows)

    # Step 3: Save CSV
    df = pd.DataFrame(all_rows)[
        ["condition", "layer", "roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi",
         "n_samples", "pct_correct"]
    ]
    os.makedirs("results/metrics", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")

    # Step 4: Print comparison at key layers
    print_comparison(results_by_condition)

    # Step 5: Compute gap across all layers and interpret
    native_df = results_by_condition.get("base_native", pd.DataFrame())
    dist_df = results_by_condition.get("base_on_distilled_text", pd.DataFrame())
    if not native_df.empty and not dist_df.empty:
        native_idx = native_df.set_index("layer")
        dist_idx = dist_df.set_index("layer")
        common = native_idx.index.intersection(dist_idx.index)
        gap_values = [
            native_idx.loc[l, "roc_auc"] - dist_idx.loc[l, "roc_auc"]
            for l in common
        ]
        print_interpretation(gap_values)


if __name__ == "__main__":
    main()
