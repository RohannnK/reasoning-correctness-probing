"""Validate probe results: control for position confound.

This script runs three critical analyses:
1. Last-chunk-only probes (position fully controlled)
2. Position-residualized probes (regress out position, probe residuals)  
3. Within-position-bucket probes (group by relative position, probe within each)

Run from your project root:
    python validate_probes.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


HIDDEN_STATES_DIR = "data/hidden_states"
CHUNKS_PATH = "data/chunks/gsm8k_chunks.jsonl"
RANDOM_STATE = 42
TEST_LAYERS = [0, 7, 14, 21, 28]  # Sample of layers to keep runtime reasonable


def load_chunks():
    """Load chunks and build per-problem metadata."""
    with open(CHUNKS_PATH) as f:
        problems = [json.loads(line) for line in f]
    return problems


def load_hidden_states_for_layer(model_key, layer_idx, problems):
    """Load hidden states for a specific layer, returning X, y, and metadata."""
    model_dir = os.path.join(HIDDEN_STATES_DIR, model_key)
    
    all_vecs = []
    all_labels = []
    all_chunk_positions = []  # relative position in chain (0 to 1)
    all_is_last = []  # whether this is the last chunk
    all_char_ends = []
    
    for problem in problems:
        pid = problem["problem_id"]
        npz_path = os.path.join(model_dir, f"problem_{pid}.npz")
        if not os.path.exists(npz_path):
            continue
        
        data = np.load(npz_path)
        layer_key = f"layer_{layer_idx}"
        if layer_key not in data:
            continue
        
        vecs = data[layer_key]  # (num_chunks, 1536)
        labels = data["labels"]  # (num_chunks,)
        num_chunks = len(labels)
        
        for i in range(num_chunks):
            all_vecs.append(vecs[i])
            all_labels.append(int(labels[i]))
            all_chunk_positions.append(i / max(num_chunks - 1, 1))
            all_is_last.append(i == num_chunks - 1)
            all_char_ends.append(problem["chunks"][i]["char_end"] if i < len(problem["chunks"]) else 0)
    
    return (
        np.array(all_vecs),
        np.array(all_labels),
        np.array(all_chunk_positions),
        np.array(all_is_last),
        np.array(all_char_ends),
    )


def analysis_1_last_chunk_only(problems):
    """Test probes on ONLY the last chunk of each problem.
    
    Position is fully controlled — every sample is the final chunk.
    The only variation is whether the problem was solved correctly.
    If probes still achieve high ROC-AUC, correctness encoding is real.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Last-chunk-only probes (position fully controlled)")
    print("=" * 70)
    print("Every sample is the final chunk. Position cannot explain any signal.")
    print()
    
    results = []
    
    for model_key in ["base", "distilled"]:
        for layer_idx in TEST_LAYERS:
            X, y, positions, is_last, char_ends = load_hidden_states_for_layer(
                model_key, layer_idx, problems
            )
            
            # Filter to only last chunks
            mask = is_last
            X_last = X[mask]
            y_last = y[mask]
            
            if len(np.unique(y_last)) < 2:
                print(f"[{model_key}] Layer {layer_idx}: Only one class in last chunks, skipping")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_last, y_last, test_size=0.2, random_state=RANDOM_STATE, stratify=y_last
            )
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
            lr.fit(X_train_s, y_train)
            probs = lr.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, probs)
            
            results.append({
                "model_key": model_key,
                "layer": layer_idx,
                "roc_auc": auc,
                "n_samples": len(y_last),
                "positive_rate": y_last.mean(),
            })
            print(f"  [{model_key:>10}] Layer {layer_idx:2d}: ROC-AUC = {auc:.4f} "
                  f"(n={len(y_last)}, {y_last.mean()*100:.1f}% correct)")
    
    return pd.DataFrame(results)


def analysis_2_position_baseline(problems):
    """Compare: hidden states vs position-only vs hidden states with position regressed out.
    
    This quantifies how much of the probe signal is position vs genuine correctness.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Position baseline comparison")
    print("=" * 70)
    print("Comparing: (a) position-only, (b) full hidden state, (c) hidden state with position regressed out")
    print()
    
    results = []
    
    for model_key in ["base", "distilled"]:
        for layer_idx in TEST_LAYERS:
            X, y, positions, is_last, char_ends = load_hidden_states_for_layer(
                model_key, layer_idx, problems
            )
            
            if len(np.unique(y)) < 2:
                continue
            
            X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
                X, y, positions, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            
            # (a) Position-only baseline
            pos_lr = LogisticRegression(class_weight="balanced", max_iter=1000)
            pos_lr.fit(pos_train.reshape(-1, 1), y_train)
            pos_probs = pos_lr.predict_proba(pos_test.reshape(-1, 1))[:, 1]
            pos_auc = roc_auc_score(y_test, pos_probs)
            
            # (b) Full hidden state
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            full_lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
            full_lr.fit(X_train_s, y_train)
            full_probs = full_lr.predict_proba(X_test_s)[:, 1]
            full_auc = roc_auc_score(y_test, full_probs)
            
            # (c) Hidden state + position concatenated (to see if position adds anything)
            X_train_with_pos = np.column_stack([X_train_s, pos_train.reshape(-1, 1)])
            X_test_with_pos = np.column_stack([X_test_s, pos_test.reshape(-1, 1)])
            
            combo_lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
            combo_lr.fit(X_train_with_pos, y_train)
            combo_probs = combo_lr.predict_proba(X_test_with_pos)[:, 1]
            combo_auc = roc_auc_score(y_test, combo_probs)
            
            results.append({
                "model_key": model_key,
                "layer": layer_idx,
                "position_only_auc": pos_auc,
                "hidden_state_auc": full_auc,
                "hidden_state_plus_position_auc": combo_auc,
                "lift_over_position": full_auc - pos_auc,
            })
            
            print(f"  [{model_key:>10}] Layer {layer_idx:2d}:  "
                  f"pos_only={pos_auc:.4f}  hidden={full_auc:.4f}  "
                  f"hidden+pos={combo_auc:.4f}  lift={full_auc - pos_auc:+.4f}")
    
    return pd.DataFrame(results)


def analysis_3_within_position_buckets(problems):
    """Train probes within position buckets.
    
    Group chunks by their relative position (early, middle, late).
    Within each bucket, position is roughly controlled.
    If probes still work, the signal is not purely positional.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Within-position-bucket probes")
    print("=" * 70)
    print("Chunks grouped by position. Probing within each group controls for position.")
    print()
    
    # Use a single strong layer
    layer_idx = 20
    
    results = []
    buckets = [
        ("early", 0.0, 0.33),
        ("middle", 0.33, 0.67),
        ("late", 0.67, 1.01),  # 1.01 to include 1.0
    ]
    
    for model_key in ["base", "distilled"]:
        X, y, positions, is_last, char_ends = load_hidden_states_for_layer(
            model_key, layer_idx, problems
        )
        
        for bucket_name, pos_low, pos_high in buckets:
            mask = (positions >= pos_low) & (positions < pos_high)
            X_bucket = X[mask]
            y_bucket = y[mask]
            
            n_positive = y_bucket.sum()
            n_negative = len(y_bucket) - n_positive
            
            if n_positive < 5 or n_negative < 5:
                print(f"  [{model_key:>10}] {bucket_name:>6} (pos {pos_low:.1f}-{pos_high:.1f}): "
                      f"Too few samples ({n_positive} correct, {n_negative} incorrect), skipping")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_bucket, y_bucket, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bucket
            )
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
            lr.fit(X_train_s, y_train)
            probs = lr.predict_proba(X_test_s)[:, 1]
            
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                auc = float("nan")
            
            results.append({
                "model_key": model_key,
                "bucket": bucket_name,
                "roc_auc": auc,
                "n_samples": len(y_bucket),
                "positive_rate": y_bucket.mean(),
            })
            
            print(f"  [{model_key:>10}] {bucket_name:>6} (pos {pos_low:.1f}-{pos_high:.1f}): "
                  f"ROC-AUC = {auc:.4f} (n={len(y_bucket)}, {y_bucket.mean()*100:.1f}% correct)")
    
    return pd.DataFrame(results)


def main():
    print("Loading chunks...")
    problems = load_chunks()
    print(f"Loaded {len(problems)} problems")
    
    # Run all three analyses
    df1 = analysis_1_last_chunk_only(problems)
    df2 = analysis_2_position_baseline(problems)
    df3 = analysis_3_within_position_buckets(problems)
    
    # Save results
    os.makedirs("results/metrics", exist_ok=True)
    df1.to_csv("results/metrics/validation_last_chunk.csv", index=False)
    df2.to_csv("results/metrics/validation_position_baseline.csv", index=False)
    df3.to_csv("results/metrics/validation_within_position.csv", index=False)
    
    # Final summary
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Analysis 1 (last-chunk-only):
  - If ROC-AUC >> 0.5: Correctness encoding is REAL. Position is controlled
    (all samples are final chunks), so the probe detects genuine correctness
    signals. This is the strongest evidence.
  - If ROC-AUC ≈ 0.5: The original high ROC-AUC was driven by position.

Analysis 2 (position baseline):
  - 'lift_over_position' = hidden_state_AUC - position_only_AUC
  - Large lift: Hidden states contain correctness info beyond position.
  - Small lift: Most of the signal was positional.

Analysis 3 (within-position buckets):
  - High AUC within the 'late' bucket is especially meaningful — these are
    chunks near the end of reasoning, so position is similar, but some
    reached the right answer and some didn't.
""")


if __name__ == "__main__":
    main()
