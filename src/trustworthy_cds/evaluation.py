"""Evaluation utilities for trained classifiers and symbolic surrogates.

The module deliberately separates two evaluation targets:

* task performance compares predictions with the historical binary outcome;
* symbolic fidelity compares a rule surrogate with its black-box reference.

Keeping these quantities separate prevents a faithful explanation from being
mistaken for an accurate clinical predictor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio, using zero when the denominator is zero."""

    return float(numerator / denominator) if denominator else 0.0


def binary_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_probability: Iterable[float] | None = None,
) -> dict[str, float | int]:
    """Compute compact and traceable metrics for a binary classifier.

    Confusion-matrix counts are included so aggregate scores can always be
    traced back to concrete observations. Probability-based metrics are added
    only when an aligned positive-class probability vector is supplied.
    """

    truth = np.asarray(list(y_true), dtype=int)
    predictions = np.asarray(list(y_pred), dtype=int)
    if truth.ndim != 1 or predictions.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(truth) == 0:
        raise ValueError("Metrics cannot be computed on an empty dataset.")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("y_pred must contain only binary labels 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "n_samples": int(len(truth)),
        "prevalence": float(np.mean(truth)),
        "predicted_positive_rate": float(np.mean(predictions)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall_sensitivity": float(
            recall_score(truth, predictions, zero_division=0)
        ),
        "specificity": _safe_ratio(tn, tn + fp),
        "negative_predictive_value": _safe_ratio(tn, tn + fn),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "f2": float(fbeta_score(truth, predictions, beta=2, zero_division=0)),
    }

    if y_probability is not None:
        probabilities = np.asarray(list(y_probability), dtype=float)
        if probabilities.ndim != 1 or len(probabilities) != len(truth):
            raise ValueError("y_probability must be one-dimensional and aligned.")
        if not np.isfinite(probabilities).all():
            raise ValueError("y_probability contains non-finite values.")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("y_probability values must lie in [0, 1].")

        metrics["roc_auc"] = (
            float(roc_auc_score(truth, probabilities))
            if len(np.unique(truth)) == 2
            else float("nan")
        )
        metrics["average_precision"] = float(
            average_precision_score(truth, probabilities)
        )

    return metrics


def confusion_table(metrics: Mapping[str, float | int]) -> pd.DataFrame:
    """Create a labelled 2x2 confusion matrix from metric counts."""

    required = (
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise KeyError(f"Confusion-matrix counts are missing: {missing}")
    return pd.DataFrame(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ],
        index=["actual 0", "actual 1"],
        columns=["predicted 0", "predicted 1"],
    )


def symbolic_surrogate_metrics(
    reference_predictions: Iterable[Any],
    symbolic_predictions: Iterable[Any],
    y_true: Iterable[int] | None = None,
    *,
    positive_label: str = "readmitted",
    negative_label: str = "not_readmitted",
) -> dict[str, float | int]:
    """Evaluate symbolic coverage, fidelity, and optional task accuracy.

    ``None`` denotes an uncovered observation. Overall fidelity treats an
    uncovered observation as a disagreement, while covered fidelity evaluates
    only rows for which the symbolic theory returned a prediction.
    """

    reference = np.asarray(list(reference_predictions), dtype=object)
    symbolic = np.asarray(list(symbolic_predictions), dtype=object)
    if reference.ndim != 1 or symbolic.ndim != 1 or len(reference) != len(symbolic):
        raise ValueError("Reference and symbolic predictions must be aligned vectors.")
    if len(reference) == 0:
        raise ValueError("Symbolic metrics cannot be computed on an empty dataset.")

    covered = np.asarray([value is not None for value in symbolic], dtype=bool)
    agreements = np.zeros(len(symbolic), dtype=bool)
    agreements[covered] = symbolic[covered] == reference[covered]
    n_covered = int(covered.sum())

    metrics: dict[str, float | int] = {
        "n_samples": int(len(reference)),
        "n_covered": n_covered,
        "coverage": float(np.mean(covered)),
        "fidelity_covered": float(np.mean(agreements[covered])) if n_covered else 0.0,
        "fidelity_overall": float(np.mean(agreements)),
    }

    if y_true is not None:
        truth = np.asarray(list(y_true), dtype=int)
        if truth.ndim != 1 or len(truth) != len(symbolic):
            raise ValueError("y_true must be aligned with symbolic predictions.")
        label_to_int = {negative_label: 0, positive_label: 1}
        symbolic_int = np.full(len(symbolic), -1, dtype=int)
        for label, integer in label_to_int.items():
            symbolic_int[symbolic == label] = integer
        recognized = covered & (symbolic_int >= 0)
        metrics["recognized_label_rate"] = float(np.mean(recognized))
        metrics["task_accuracy_covered"] = (
            float(accuracy_score(truth[recognized], symbolic_int[recognized]))
            if recognized.any()
            else 0.0
        )
        task_correct = np.zeros(len(truth), dtype=bool)
        task_correct[recognized] = symbolic_int[recognized] == truth[recognized]
        metrics["task_accuracy_overall"] = float(np.mean(task_correct))

    return metrics


def class_conditional_fidelity_table(
    black_box_labels: Iterable[Any],
    symbolic_labels: Iterable[Any],
    *,
    labels: tuple[str, ...] = ("not_readmitted", "readmitted"),
) -> pd.DataFrame:
    """Report symbolic coverage and fidelity within each black-box class."""

    reference = np.asarray(list(black_box_labels), dtype=object)
    symbolic = np.asarray(list(symbolic_labels), dtype=object)
    if reference.shape != symbolic.shape or reference.ndim != 1:
        raise ValueError("Black-box and symbolic labels must be aligned vectors.")

    covered = np.asarray([value is not None for value in symbolic], dtype=bool)
    records: list[dict[str, float | int | str]] = []
    for label in labels:
        mask = reference == label
        n_cases = int(mask.sum())
        records.append(
            {
                "black_box_class": label,
                "n_black_box_predictions": n_cases,
                "symbolic_coverage": (
                    float(covered[mask].mean()) if n_cases else float("nan")
                ),
                "class_conditional_fidelity": (
                    float((symbolic[mask] == reference[mask]).mean())
                    if n_cases
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(records).set_index("black_box_class")


def symbolic_task_metrics_on_recognized(
    y_true: Iterable[int],
    symbolic_labels: Iterable[Any],
    *,
    positive_label: str = "readmitted",
    negative_label: str = "not_readmitted",
) -> dict[str, float | int]:
    """Evaluate task performance only where a symbolic label is recognized."""

    truth = np.asarray(list(y_true), dtype=int)
    symbolic = np.asarray(list(symbolic_labels), dtype=object)
    if truth.shape != symbolic.shape or truth.ndim != 1:
        raise ValueError("y_true and symbolic_labels must be aligned vectors.")

    mapping = {negative_label: 0, positive_label: 1}
    recognized = np.asarray([label in mapping for label in symbolic], dtype=bool)
    if not recognized.any():
        return {"recognized_rows": 0, "recognized_rate": 0.0}

    predictions = np.asarray(
        [mapping[label] for label in symbolic[recognized]],
        dtype=int,
    )
    metrics = binary_classification_metrics(truth[recognized], predictions)
    metrics["recognized_rows"] = int(recognized.sum())
    metrics["recognized_rate"] = float(recognized.mean())
    return metrics