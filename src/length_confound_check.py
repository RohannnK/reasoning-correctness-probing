"""Length confound check: can chain-of-thought length alone predict correctness?

Trains a logistic regression on structural features (total_chunks, total_char_length,
last_chunk_char_position) with NO hidden states, using the same GroupShuffleSplit
methodology as train_probes. If this baseline matches the probe AUC (~0.855),
the probe signal is confounded by length.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.utils import load_jsonl

CHUNKS_PATH = "data/chunks/gsm8k_chunks.jsonl"
OUTPUT_DIR = "results/metrics"
N_SPLITS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    problems = load_jsonl(CHUNKS_PATH)
    print(f"Loaded {len(problems)} problems")

    features = []
    labels = []
    groups = []

    for p in problems:
        chunks = p["chunks"]
        if not chunks:
            continue
        total_chunks = len(chunks)
        total_char_length = len(p["full_output"])
        last_chunk_char_pos = chunks[-1]["char_start"]
        label = int(chunks[-1]["is_correct"])

        features.append([total_chunks, total_char_length, last_chunk_char_pos])
        labels.append(label)
        groups.append(p["problem_id"])

    X = np.array(features, dtype=np.float64)
    y = np.array(labels)
    g = np.array(groups)

    print(f"Samples: {len(y)} (one per problem, last-chunk-only)")
    print(f"Positive rate: {y.mean():.3f}")
    print(f"Features: total_chunks, total_char_length, last_chunk_char_position")

    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    aucs = []

    for train_idx, test_idx in gss.split(X, y, groups=g):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
        )
        lr.fit(X_tr_s, y_tr)
        probs = lr.predict_proba(X_te_s)[:, 1]
        aucs.append(roc_auc_score(y_te, probs))

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    ci_lo = mean_auc - 1.96 * std_auc
    ci_hi = mean_auc + 1.96 * std_auc

    probe_base = 0.8578
    probe_dist = 0.8555
    pos_baseline = 0.8496
    delta_base = probe_base - mean_auc
    delta_dist = probe_dist - mean_auc

    print(f"\n{'='*60}")
    print("LENGTH-ONLY CONFOUND CHECK")
    print(f"{'='*60}")
    print(f"\nLength-only AUC: {mean_auc:.4f} ± {1.96*std_auc:.4f}  "
          f"(10-split mean ± 1.96·std)")
    print(f"  CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\nComparison:")
    print(f"  Position-only baseline:  {pos_baseline:.4f}")
    print(f"  Length-only baseline:    {mean_auc:.4f}")
    print(f"  Best probe (base L17):   {probe_base:.4f}")
    print(f"  Best probe (dist L19):   {probe_dist:.4f}")
    print(f"  Δ(base probe − length):  {delta_base:+.4f}")
    print(f"  Δ(dist probe − length):  {delta_dist:+.4f}")

    if mean_auc >= 0.855:
        interp = ("LENGTH CONFOUND: length-only baseline matches probe AUC — "
                   "probe signal may be largely explained by chain length.")
    elif mean_auc >= 0.840:
        interp = ("PARTIAL CONFOUND: length-only baseline is close to probe AUC — "
                   "some signal may come from length, but probes capture additional information.")
    else:
        interp = ("GENUINE SIGNAL: length-only baseline is well below probe AUC — "
                   "hidden states carry correctness information beyond structural features.")
    print(f"\nInterpretation: {interp}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "length_confound_check.csv")
    pd.DataFrame([{
        "features": "total_chunks,total_char_length,last_chunk_char_position",
        "n_samples": len(y),
        "positive_rate": float(y.mean()),
        "n_splits": len(aucs),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "probe_base_best_auc": probe_base,
        "probe_dist_best_auc": probe_dist,
        "position_baseline_auc": pos_baseline,
        "delta_base": delta_base,
        "delta_dist": delta_dist,
    }]).to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
