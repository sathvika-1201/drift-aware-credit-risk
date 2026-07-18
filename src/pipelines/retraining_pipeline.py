import argparse
import json
import shutil
import time

import joblib
import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task

from src.config import (
    CHALLENGER_MODEL_PATH,
    CHAMPION_MODEL_PATH,
    CURRENT_DATA_END_DATE,
    CURRENT_DATA_START_DATE,
    DATE_COLUMN,
    DRIFT_REFERENCE_END_DATE,
    DRIFT_SAMPLE_ROWS,
    DRIFT_THRESHOLD,
    EVALUATION_SAMPLE_ROWS,
    LEGACY_MODEL_PATH,
    MAX_PR_AUC_DROP_ALLOWED,
    MAX_ROC_AUC_DROP_ALLOWED,
    MIN_RECALL_IMPROVEMENT,
    PROCESSED_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.features.build_features import SELECTED_FEATURES, prepare_features
from src.models.train_model import evaluate_pipeline, train_model




def load_drift_samples(
    path=PROCESSED_DATA_PATH,
    chunksize=100_000,
    max_reference_rows=DRIFT_SAMPLE_ROWS,
    max_current_rows=DRIFT_SAMPLE_ROWS,
):
    reference_chunks = []
    current_chunks = []

    reference_rows = 0
    current_rows = 0

    required_columns = set(SELECTED_FEATURES + [DATE_COLUMN, TARGET_COLUMN])

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=True,
            usecols=lambda column: column in required_columns,
        ),
        start=1,
    ):
        print(f"Reading chunk {chunk_number}...")

        chunk[DATE_COLUMN] = pd.to_datetime(chunk[DATE_COLUMN], errors="coerce")

        reference_chunk = chunk[chunk[DATE_COLUMN] < DRIFT_REFERENCE_END_DATE]

        current_mask = chunk[DATE_COLUMN] >= CURRENT_DATA_START_DATE
        if CURRENT_DATA_END_DATE:
            current_mask &= chunk[DATE_COLUMN] < CURRENT_DATA_END_DATE

        current_chunk = chunk[current_mask]

        if reference_rows < max_reference_rows and not reference_chunk.empty:
            needed = max_reference_rows - reference_rows
            selected_reference = reference_chunk.head(needed)
            reference_chunks.append(selected_reference)
            reference_rows += len(selected_reference)

        if current_rows < max_current_rows and not current_chunk.empty:
            needed = max_current_rows - current_rows
            selected_current = current_chunk.head(needed)
            current_chunks.append(selected_current)
            current_rows += len(selected_current)

        print(f"Collected reference={reference_rows:,}, current={current_rows:,}")

        if reference_rows >= max_reference_rows and current_rows >= max_current_rows:
            break

    if not reference_chunks:
        raise ValueError(
            f"No reference rows found before {DRIFT_REFERENCE_END_DATE}."
        )

    if not current_chunks:
        raise ValueError(
            f"No current rows found from {CURRENT_DATA_START_DATE} onward."
        )

    reference = pd.concat(reference_chunks, ignore_index=True)
    current = pd.concat(current_chunks, ignore_index=True)

    return reference, current


def extract_dataset_drift_score(report_result) -> float:
    result_dict = report_result.dict()

    drifted_columns = 0
    total_columns = 0

    def search(obj):
        nonlocal drifted_columns, total_columns

        if isinstance(obj, dict):
            if "number_of_drifted_columns" in obj:
                drifted_columns = obj.get(
                    "number_of_drifted_columns",
                    drifted_columns,
                )

            if "number_of_columns" in obj:
                total_columns = obj.get(
                    "number_of_columns",
                    total_columns,
                )

            for value in obj.values():
                search(value)

        elif isinstance(obj, list):
            for item in obj:
                search(item)

    search(result_dict)

    if total_columns == 0:
        print("Could not find drift column counts in Evidently output.")
        return 0.0

    return drifted_columns / total_columns


@task
def check_drift():
    reference, current = load_drift_samples()

    reference = reference.drop(columns=[TARGET_COLUMN], errors="ignore")
    current = current.drop(columns=[TARGET_COLUMN], errors="ignore")

    reference = prepare_features(reference)
    current = prepare_features(current)

    common_columns = reference.columns.intersection(current.columns)
    reference = reference[common_columns]
    current = current[common_columns]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    html_path = REPORTS_DIR / "retraining_drift_report.html"
    json_path = REPORTS_DIR / "retraining_drift_summary.json"

    result.save_html(str(html_path))

    drift_score = extract_dataset_drift_score(result)
    drift_detected = drift_score >= DRIFT_THRESHOLD

    summary = {
        "drift_score": drift_score,
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
        "html_report": str(html_path),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    mlflow.set_experiment("drift-aware-credit-risk")

    with mlflow.start_run(run_name="drift-check"):
        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
        mlflow.log_param("drift_detected", drift_detected)
        mlflow.log_artifact(str(html_path))
        mlflow.log_artifact(str(json_path))

    print(f"Drift score: {drift_score:.2%}")
    print(f"Drift threshold: {DRIFT_THRESHOLD:.2%}")
    print(f"Drift detected: {drift_detected}")

    return drift_detected


def load_recent_holdout():
    _, holdout_df = load_drift_samples(
        max_reference_rows=1,
        max_current_rows=EVALUATION_SAMPLE_ROWS,
    )

    X_holdout = holdout_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    y_holdout = holdout_df[TARGET_COLUMN].astype(int)

    X_holdout = prepare_features(X_holdout)

    return X_holdout, y_holdout


@task
def train_challenger_model(drift_detected: bool, force_retrain: bool = False):
    if not drift_detected and not force_retrain:
        print("No significant drift detected. Challenger training skipped.")
        return False

    print("Drift detected. Training challenger model...")

    train_model(
        save_path=CHALLENGER_MODEL_PATH,
        run_name="challenger-training-after-drift",
        train_start_date=None,
        train_end_date=CURRENT_DATA_START_DATE,
        validation_start_date=CURRENT_DATA_START_DATE,
        validation_end_date=CURRENT_DATA_END_DATE,
    )

    print(f"Challenger model saved to {CHALLENGER_MODEL_PATH}")

    return True


def promote_challenger():
    """Atomically replace the serving model so readers never see a partial file."""
    temporary_path = CHAMPION_MODEL_PATH.with_suffix(".tmp")
    shutil.copy2(CHALLENGER_MODEL_PATH, temporary_path)
    temporary_path.replace(CHAMPION_MODEL_PATH)


@task
def compare_champion_and_challenger(challenger_trained: bool):
    if not challenger_trained:
        return {
            "promotion_decision": "skipped",
            "reason": "No challenger was trained because drift was not detected.",
        }

    if not CHAMPION_MODEL_PATH.exists() and LEGACY_MODEL_PATH.exists():
        CHAMPION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEGACY_MODEL_PATH, CHAMPION_MODEL_PATH)
        print("Initialized champion from the existing legacy model.")

    if not CHAMPION_MODEL_PATH.exists():
        print("Champion model not found. Promoting challenger as first champion.")
        promote_challenger()

        return {
            "promotion_decision": "promoted",
            "reason": "No existing champion model found.",
        }

    champion_model = joblib.load(CHAMPION_MODEL_PATH)
    challenger_model = joblib.load(CHALLENGER_MODEL_PATH)

    X_holdout, y_holdout = load_recent_holdout()

    champion_metrics = evaluate_pipeline(
        champion_model,
        X_holdout,
        y_holdout,
    )

    challenger_metrics = evaluate_pipeline(
        challenger_model,
        X_holdout,
        y_holdout,
    )

    champion_metrics = {key: float(value) for key, value in champion_metrics.items()}
    challenger_metrics = {key: float(value) for key, value in challenger_metrics.items()}

    pr_auc_gate = (
        challenger_metrics["pr_auc"]
        >= champion_metrics["pr_auc"] - MAX_PR_AUC_DROP_ALLOWED
    )

    roc_auc_gate = (
        challenger_metrics["roc_auc"]
        >= champion_metrics["roc_auc"] - MAX_ROC_AUC_DROP_ALLOWED
    )

    recall_gate = (
        challenger_metrics["recall"]
        >= champion_metrics["recall"] + MIN_RECALL_IMPROVEMENT
    )

    promote = pr_auc_gate and roc_auc_gate and recall_gate

    decision = {
        "promotion_decision": "promoted" if promote else "rejected",
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "promotion_gates": {
            "pr_auc_gate": pr_auc_gate,
            "roc_auc_gate": roc_auc_gate,
            "recall_gate": recall_gate,
        },
        "rules": {
            "max_pr_auc_drop_allowed": MAX_PR_AUC_DROP_ALLOWED,
            "max_roc_auc_drop_allowed": MAX_ROC_AUC_DROP_ALLOWED,
            "min_recall_improvement": MIN_RECALL_IMPROVEMENT,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    decision_path = REPORTS_DIR / "promotion_decision.json"

    with open(decision_path, "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=4)

    mlflow.set_experiment("drift-aware-credit-risk")

    with mlflow.start_run(run_name="champion-challenger-comparison"):
        for metric_name, metric_value in champion_metrics.items():
            mlflow.log_metric(f"champion_{metric_name}", metric_value)

        for metric_name, metric_value in challenger_metrics.items():
            mlflow.log_metric(f"challenger_{metric_name}", metric_value)

        mlflow.log_param("promotion_decision", decision["promotion_decision"])
        mlflow.log_param("pr_auc_gate", pr_auc_gate)
        mlflow.log_param("roc_auc_gate", roc_auc_gate)
        mlflow.log_param("recall_gate", recall_gate)
        mlflow.log_artifact(str(decision_path))

    print("Champion metrics:")
    for metric_name, metric_value in champion_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("Challenger metrics:")
    for metric_name, metric_value in challenger_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print(f"Promotion decision: {decision['promotion_decision']}")

    if promote:
        promote_challenger()
        print("Challenger promoted to champion.")
    else:
        print("Challenger rejected. Existing champion kept.")

    return decision


@flow(name="champion-challenger-retraining-pipeline")
def champion_challenger_retraining_pipeline(force_retrain: bool = False):
    drift_detected = check_drift()
    challenger_trained = train_challenger_model(
        drift_detected, force_retrain=force_retrain
    )
    decision = compare_champion_and_challenger(challenger_trained)

    print("Final retraining decision:")
    print(decision)

    return decision


if __name__ == "__main__":
    champion_challenger_retraining_pipeline()
