# evaluation/metrics.py

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import pandas as pd


def compute_metrics(
    df: pd.DataFrame,
    truth_col: str = "is_true_anomaly",
    pred_col: str = "anomaly_flag",
    verbose: bool = True
):
    y_true = df[truth_col]
    y_pred = df[pred_col]

    precision = precision_score(y_true, y_pred, zero_division=0)

    recall = recall_score(y_true, y_pred, zero_division=0)

    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    if verbose:
        print("=== EVALUATION METRICS ===")

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(report)

    return metrics