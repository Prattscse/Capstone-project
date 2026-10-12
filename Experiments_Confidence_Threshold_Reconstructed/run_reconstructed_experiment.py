#!/usr/bin/env python3
"""Reconstructed CICIDS2017 vote-agreement / deferral experiment for L2DHF.

Important: this is NOT an exact reproduction of the authors' trained models. The
GitHub ZIP contains Git LFS pointers for the processed datasets and does not
include the fitted PCA transform. This script reconstructs a documented
preprocessing pipeline from the raw MachineLearningCSV.zip and trains fresh
classifiers. It should be described as a reconstructed extension in any report.
"""
from __future__ import annotations
import argparse, json, sys, time, zipfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

SEED = 42
PRIORITY_CLASSES = [0, 2, 3, 4]
ENCODE = {priority: i for i, priority in enumerate(PRIORITY_CLASSES)}
DECODE = {v: k for k, v in ENCODE.items()}
THRESHOLDS = [0.50, 0.75, 1.00]
DUPLICATE_FEATURE = "Fwd Header Length.1"


def map_priority(label: str) -> int | None:
    """Map raw attack labels to reconstructed severity categories.

    The mapping is reconstructed from the paper's description of assigning
    CVSS scores to CICIDS2017 attack families and mapping CVSS v3.x bands to
    severity classes. This is an explicit assumption, not recovered author
    metadata.
    """
    s = str(label).strip().lower()
    if s == "benign":
        return 0
    if "bot" in s:
        return 4
    if "ddos" in s or "heartbleed" in s or "sql injection" in s:
        return 3
    if (s.startswith("dos ") or s == "portscan" or "patator" in s
            or "web attack" in s or "infiltration" in s):
        return 2
    return None


def reservoir_sample_from_zip(zip_path: Path, cap_per_class: int, chunksize: int, seed: int):
    rng = np.random.default_rng(seed)
    reservoirs: dict[int, pd.DataFrame] = {p: pd.DataFrame() for p in PRIORITY_CLASSES}
    seen = {p: 0 for p in PRIORITY_CLASSES}
    total_rows = valid_rows = invalid_rows = unmapped_rows = 0
    feature_cols = None
    start = time.time()

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV files found in the provided ZIP archive.")
        for csv_name in csv_names:
            print(f"Reading {csv_name}", flush=True)
            with zf.open(csv_name) as stream:
                for chunk in pd.read_csv(stream, chunksize=chunksize, low_memory=False):
                    total_rows += len(chunk)
                    chunk.columns = [str(c).strip() for c in chunk.columns]
                    if "Label" not in chunk.columns:
                        raise ValueError(f"Could not find a Label column in {csv_name}.")
                    if feature_cols is None:
                        feature_cols = [c for c in chunk.columns if c != "Label" and c != DUPLICATE_FEATURE]
                    missing_cols = [c for c in feature_cols if c not in chunk.columns]
                    if missing_cols:
                        raise ValueError(f"Feature columns missing from {csv_name}: {missing_cols}")
                    labels = chunk["Label"].astype(str).str.strip()
                    mapped = labels.map(map_priority)
                    unmapped_rows += int(mapped.isna().sum())
                    keep = mapped.notna()
                    if not keep.any():
                        invalid_rows += len(chunk)
                        continue
                    x = chunk.loc[keep, feature_cols].apply(pd.to_numeric, errors="coerce")
                    x = x.replace([np.inf, -np.inf], np.nan)
                    finite = np.isfinite(x.to_numpy(dtype=np.float64, copy=False)).all(axis=1)
                    invalid_rows += int((~finite).sum())
                    if not finite.any():
                        continue
                    x = x.loc[finite].reset_index(drop=True).astype(np.float32)
                    y = mapped.loc[keep].loc[finite].astype(int).reset_index(drop=True)
                    valid_rows += len(x)

                    for priority in PRIORITY_CLASSES:
                        idx = np.flatnonzero(y.to_numpy() == priority)
                        if len(idx) == 0:
                            continue
                        seen[priority] += len(idx)
                        fresh = x.iloc[idx].copy()
                        fresh["__random_key__"] = rng.random(len(fresh))
                        old = reservoirs[priority]
                        candidates = pd.concat([old, fresh], ignore_index=True) if len(old) else fresh
                        reservoirs[priority] = candidates.nsmallest(cap_per_class, "__random_key__").reset_index(drop=True)

    frames = []
    for priority in PRIORITY_CLASSES:
        frame = reservoirs[priority].drop(columns=["__random_key__"]).copy()
        frame["priority"] = priority
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    stats = {
        "source_zip": str(zip_path),
        "source_csv_files": len(csv_names),
        "raw_rows_read": int(total_rows),
        "valid_rows_after_finite_numeric_filter": int(valid_rows),
        "invalid_or_unusable_rows_skipped": int(invalid_rows),
        "unmapped_label_rows": int(unmapped_rows),
        "clean_valid_counts_by_priority_before_sampling": {str(k): int(v) for k, v in seen.items()},
        "sample_counts_by_priority": {str(k): int((data.priority == k).sum()) for k in PRIORITY_CLASSES},
        "cap_per_priority": cap_per_class,
        "elapsed_sampling_seconds": round(time.time() - start, 2),
        "feature_columns_before_constant_filter": int(len(feature_cols or [])),
        "feature_columns": feature_cols,
    }
    return data, stats


def majority_with_first_tie(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds, agreements = [], []
    for row in votes:
        labels, counts = np.unique(row, return_counts=True)
        max_count = int(counts.max())
        tied = set(labels[counts == max_count].tolist())
        winner = next(int(v) for v in row if int(v) in tied)
        preds.append(winner)
        agreements.append(max_count / len(row))
    return np.asarray(preds, dtype=int), np.asarray(agreements, dtype=float)


def evaluate_thresholds(y_true_priority, y_pred_priority, agreement, output_dir: Path):
    baseline_accuracy = accuracy_score(y_true_priority, y_pred_priority)
    rows = []
    for threshold in THRESHOLDS:
        defer = agreement < threshold
        auto = ~defer
        auto_count = int(auto.sum())
        deferred_count = int(defer.sum())
        high_critical = np.isin(y_true_priority, [3, 4])
        severe_underpriority = (y_pred_priority < y_true_priority) & high_critical
        severe_auto = high_critical & auto
        rows.append({
            "agreement_threshold": threshold,
            "test_alerts": int(len(y_true_priority)),
            "majority_vote_baseline_accuracy_percent": round(100 * float(baseline_accuracy), 4),
            "majority_vote_macro_f1": round(float(f1_score(y_true_priority, y_pred_priority, labels=PRIORITY_CLASSES, average="macro", zero_division=0)), 4),
            "deferred_to_analyst_count": deferred_count,
            "deferred_to_analyst_percent": round(100 * float(defer.mean()), 4),
            "auto_processed_count": auto_count,
            "auto_processed_percent": round(100 * float(auto.mean()), 4),
            "auto_processed_accuracy_percent": round(100 * float(accuracy_score(y_true_priority[auto], y_pred_priority[auto])), 4) if auto_count else np.nan,
            "deferred_accuracy_percent_for_analysis_only": round(100 * float(accuracy_score(y_true_priority[defer], y_pred_priority[defer])), 4) if deferred_count else np.nan,
            "high_critical_auto_processed_count": int(severe_auto.sum()),
            "high_critical_underprioritized_auto_count": int(severe_underpriority[auto].sum()),
            "high_critical_underprioritization_rate_among_auto_percent": round(100 * float(severe_underpriority[auto].sum() / severe_auto.sum()), 4) if severe_auto.sum() else np.nan,
        })
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "confidence_threshold_results.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, default=Path("MachineLearningCSV.zip"), help="Path to CICIDS2017 MachineLearningCSV.zip (default: MachineLearningCSV.zip in current directory)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for outputs (default: ./results)")
    parser.add_argument("--cap-per-class", type=int, default=20000, help="Maximum sample per mapped priority class (default: 20000)")
    parser.add_argument("--chunksize", type=int, default=100000, help="CSV read chunk size")
    args = parser.parse_args()
    if not args.dataset_zip.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {args.dataset_zip}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Step 1/5: Clean raw CICIDS2017 CSVs and form a class-stratified sample...", flush=True)
    data, meta = reservoir_sample_from_zip(args.dataset_zip, args.cap_per_class, args.chunksize, SEED)
    if meta["unmapped_label_rows"]:
        print(f"Warning: {meta['unmapped_label_rows']} rows had unrecognised labels and were excluded.", flush=True)
    feature_cols = [c for c in data.columns if c != "priority"]
    variable_cols = [c for c in feature_cols if data[c].nunique(dropna=False) > 1]
    dropped_constant = sorted(set(feature_cols) - set(variable_cols))
    X = data[variable_cols].astype(np.float32)
    y_priority = data["priority"].astype(int).to_numpy()
    y_encoded = np.array([ENCODE[int(v)] for v in y_priority], dtype=int)

    meta["constant_features_removed_on_sample"] = dropped_constant
    meta["model_input_feature_count_before_PCA"] = len(variable_cols)
    indices = np.arange(len(data))
    i_train, i_test = train_test_split(indices, test_size=0.25, random_state=SEED, stratify=y_encoded)
    X_train, X_test = X.iloc[i_train], X.iloc[i_test]
    y_train, y_test_encoded = y_encoded[i_train], y_encoded[i_test]
    y_test_priority = y_priority[i_test]

    print("Step 2/5: Fit MinMax normalization and 12-component PCA on training data...", flush=True)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=12, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    meta["pca_components"] = 12
    meta["pca_explained_variance_ratio_sum_on_training_sample"] = float(pca.explained_variance_ratio_.sum())
    meta["pca_explained_variance_ratio_by_component"] = pca.explained_variance_ratio_.tolist()
    meta["train_rows"] = int(len(i_train))
    meta["test_rows"] = int(len(i_test))
    meta["training_priority_counts"] = {str(k): int((y_priority[i_train] == k).sum()) for k in PRIORITY_CLASSES}
    meta["test_priority_counts"] = {str(k): int((y_test_priority == k).sum()) for k in PRIORITY_CLASSES}
    meta["priority_mapping"] = {
        "0": "Normal / benign traffic",
        "2": "Medium: CVSS score 4.0-6.9 (DoS, scans, Patator, web brute force/XSS, infiltration)",
        "3": "High: CVSS score 7.0-8.9 (DDoS, Heartbleed, SQL injection)",
        "4": "Critical: CVSS score 9.0-10.0 (Bot)",
    }
    meta["limitations"] = [
        "This is a reconstructed experiment, not a faithful reproduction of the authors' original trained-model results.",
        "The original GitHub ZIP contains Git LFS pointers instead of the five processed CSV files; the fitted PCA transform is not available in the supplied files.",
        "CVSS-to-priority labels were reconstructed from the paper's description and cited CVSS score table; they may differ from the authors' exact row-level annotation.",
        "Only a capped random sample per priority class was used for manageable runtime; class-balanced test metrics do not represent natural SOC traffic prevalence.",
        "Vote agreement is not a calibrated probability and threshold evaluation does not simulate the full RL agent or real analyst outcomes.",
    ]

    print("Step 3/5: Train the four ensemble-style classifiers on the reconstructed PCA features...", flush=True)
    model_specs = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED), n_estimators=50, learning_rate=0.5, random_state=SEED),
        "XGBoost": None,
        "NaiveBayes": GaussianNB(var_smoothing=7.50886e-06),
    }
    try:
        from xgboost import XGBClassifier
        model_specs["XGBoost"] = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=SEED)
    except Exception as exc:
        raise RuntimeError(f"XGBoost is required for the four-model experiment: {exc}") from exc

    predictions_encoded = []
    individual_rows = []
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for name, model in model_specs.items():
        t0 = time.time()
        model.fit(X_train_pca, y_train)
        pred_enc = np.asarray(model.predict(X_test_pca), dtype=int)
        pred_priority = np.array([DECODE[int(v)] for v in pred_enc], dtype=int)
        predictions_encoded.append(pred_priority)
        individual_rows.append({
            "model": name,
            "accuracy_percent": round(100 * float(accuracy_score(y_test_priority, pred_priority)), 4),
            "macro_f1": round(float(f1_score(y_test_priority, pred_priority, labels=PRIORITY_CLASSES, average="macro", zero_division=0)), 4),
            "training_seconds": round(time.time() - t0, 2),
        })
        joblib.dump(model, model_dir / f"{name.lower()}_reconstructed.joblib")
        print(f"  {name}: accuracy={individual_rows[-1]['accuracy_percent']:.2f}%, macro-F1={individual_rows[-1]['macro_f1']:.4f}", flush=True)
    pd.DataFrame(individual_rows).to_csv(args.output_dir / "individual_model_metrics.csv", index=False)

    print("Step 4/5: Evaluate majority voting and agreement thresholds...", flush=True)
    votes = np.column_stack(predictions_encoded)
    ensemble_predictions, agreement = majority_with_first_tie(votes)
    results = evaluate_thresholds(y_test_priority, ensemble_predictions, agreement, args.output_dir)
    print("\nConfidence-threshold results:")
    print(results.to_string(index=False))
    print("\nPCA variance explained by 12 components: {:.3f}%".format(100 * meta["pca_explained_variance_ratio_sum_on_training_sample"]))

    pred_df = pd.DataFrame({
        "true_priority": y_test_priority,
        "random_forest_priority": votes[:, 0],
        "adaboost_priority": votes[:, 1],
        "xgboost_priority": votes[:, 2],
        "naive_bayes_priority": votes[:, 3],
        "majority_vote_priority": ensemble_predictions,
        "vote_agreement": agreement,
    })
    pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False)
    report = classification_report(y_test_priority, ensemble_predictions, labels=PRIORITY_CLASSES,
                                   target_names=["Normal", "Medium", "High", "Critical"], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(args.output_dir / "ensemble_classification_report.csv")

    print("Step 5/5: Save preprocessing objects and run metadata...", flush=True)
    joblib.dump(scaler, model_dir / "minmax_scaler_reconstructed.joblib")
    joblib.dump(pca, model_dir / "pca_12_reconstructed.joblib")
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    data.iloc[:, :-1].head(0).to_csv(args.output_dir / "sampled_data_schema.csv", index=False)
    print(f"\nSaved experiment outputs to {args.output_dir.resolve()}", flush=True)
    print("Caveat: results are for this reconstructed pipeline and sample, not the original authors' published results.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
