"""Train linear and MLP probes on hidden states to detect correctness signals."""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.utils import compute_ece, load_hidden_states


def train_probes(
    hidden_states_dir: str,
    model_short_name: str,
    output_dir: str,
    C_values: list[float] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train LogisticRegression and MLP probes for each layer.

    Args:
        hidden_states_dir: Directory containing npz files.
        model_short_name: Model short name for file matching.
        output_dir: Directory to save probe checkpoints and results.
        C_values: Regularization values for grid search.
        test_size: Fraction of data for test set.
        random_state: Random seed.

    Returns:
        DataFrame with probe results per layer.
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    print(f"Loading hidden states for {model_short_name}...")
    layer_data = load_hidden_states(hidden_states_dir, model_short_name)

    if not layer_data:
        print("No hidden states found.")
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for layer_idx, (X, y) in sorted(layer_data.items()):
        print(f"\nLayer {layer_idx}: {X.shape[0]} samples, {X.shape[1]} features, "
              f"positive rate: {y.mean():.3f}")

        if len(np.unique(y)) < 2:
            print(f"  Skipping layer {layer_idx}: only one class present.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Logistic Regression with grid search ---
        lr_cv = GridSearchCV(
            LogisticRegression(max_iter=1000, solver="lbfgs"),
            param_grid={"C": C_values},
            scoring="roc_auc",
            cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=random_state),
            n_jobs=-1,
        )
        lr_cv.fit(X_train_scaled, y_train)
        lr_best = lr_cv.best_estimator_

        lr_probs = lr_best.predict_proba(X_test_scaled)[:, 1]
        lr_preds = lr_best.predict(X_test_scaled)

        lr_metrics = {
            "model": model_short_name,
            "layer": layer_idx,
            "probe_type": "logistic_regression",
            "best_C": lr_cv.best_params_["C"],
            "roc_auc": roc_auc_score(y_test, lr_probs),
            "ece": compute_ece(y_test, lr_probs),
            "precision": precision_score(y_test, lr_preds, zero_division=0),
            "recall": recall_score(y_test, lr_preds, zero_division=0),
            "f1": f1_score(y_test, lr_preds, zero_division=0),
        }
        results.append(lr_metrics)

        # Save probe checkpoint
        joblib.dump(
            {"probe": lr_best, "scaler": scaler},
            os.path.join(output_dir, f"{model_short_name}_lr_layer_{layer_idx}.joblib"),
        )

        # --- MLP probe ---
        mlp = MLPClassifier(
            hidden_layer_sizes=(256,),
            early_stopping=True,
            max_iter=500,
            random_state=random_state,
            validation_fraction=0.15,
        )
        mlp.fit(X_train_scaled, y_train)

        mlp_probs = mlp.predict_proba(X_test_scaled)[:, 1]
        mlp_preds = mlp.predict(X_test_scaled)

        mlp_metrics = {
            "model": model_short_name,
            "layer": layer_idx,
            "probe_type": "mlp",
            "best_C": None,
            "roc_auc": roc_auc_score(y_test, mlp_probs),
            "ece": compute_ece(y_test, mlp_probs),
            "precision": precision_score(y_test, mlp_preds, zero_division=0),
            "recall": recall_score(y_test, mlp_preds, zero_division=0),
            "f1": f1_score(y_test, mlp_preds, zero_division=0),
        }
        results.append(mlp_metrics)

        joblib.dump(
            {"probe": mlp, "scaler": scaler},
            os.path.join(output_dir, f"{model_short_name}_mlp_layer_{layer_idx}.joblib"),
        )

        print(f"  LR  AUC={lr_metrics['roc_auc']:.4f}  F1={lr_metrics['f1']:.4f}")
        print(f"  MLP AUC={mlp_metrics['roc_auc']:.4f}  F1={mlp_metrics['f1']:.4f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{model_short_name}_probe_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probes on hidden states")
    parser.add_argument("--hidden_states_dir", type=str, required=True)
    parser.add_argument("--model_short_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    train_probes(
        hidden_states_dir=args.hidden_states_dir,
        model_short_name=args.model_short_name,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
