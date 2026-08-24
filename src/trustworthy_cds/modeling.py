"""Minimal model-loading and thresholding utilities for the final analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def positive_class_probability(
    predictor: Any,
    X: pd.DataFrame,
    positive_label: int = 1,
) -> np.ndarray:
    """Return the positive-class probability without assuming class order."""

    if not hasattr(predictor, "predict_proba") or not hasattr(predictor, "classes_"):
        raise TypeError("The predictor must expose predict_proba() and classes_.")

    classes = np.asarray(predictor.classes_)
    matching = np.flatnonzero(classes == positive_label)
    if len(matching) != 1:
        raise ValueError(
            f"Positive label {positive_label!r} is not uniquely present in classes_."
        )

    probabilities = np.asarray(predictor.predict_proba(X), dtype=float)
    expected_shape = (len(X), len(classes))
    if probabilities.shape != expected_shape:
        raise ValueError(
            "predict_proba() returned shape "
            f"{probabilities.shape}, expected {expected_shape}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("predict_proba() returned non-finite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("predict_proba() returned values outside [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each predict_proba() row must sum to one.")
    return probabilities[:, int(matching[0])]


class ThresholdedClassifier:
    """Apply an explicit threshold to any fitted probabilistic classifier.

    ``predict_proba`` is delegated to the original estimator. ``predict`` uses
    the stored positive-class probability and the configured threshold, making
    the exact decision rule visible to PSyKE and to the evaluation code.
    """

    def __init__(
        self,
        predictor: Any,
        threshold: float = 0.5,
        positive_label: int = 1,
        negative_label: int = 0,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1].")
        if not hasattr(predictor, "predict_proba") or not hasattr(
            predictor, "classes_"
        ):
            raise TypeError(
                "The wrapped predictor must expose predict_proba() and classes_."
            )

        classes = np.asarray(predictor.classes_)
        if set(classes.tolist()) != {negative_label, positive_label}:
            raise ValueError(
                "The wrapped classifier must contain exactly the configured "
                "binary labels."
            )

        self.predictor = predictor
        self.threshold = float(threshold)
        self.positive_label = positive_label
        self.negative_label = negative_label

    @property
    def classes_(self) -> np.ndarray:
        """Expose the class order of the wrapped estimator."""

        return np.asarray(self.predictor.classes_)

    @property
    def feature_names_in_(self) -> np.ndarray | None:
        """Expose the fitted feature contract when the estimator records it."""

        return getattr(self.predictor, "feature_names_in_", None)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Delegate probability prediction to the fitted estimator."""

        return np.asarray(self.predictor.predict_proba(X), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Convert positive-class probabilities into thresholded labels."""

        probabilities = positive_class_probability(self, X, self.positive_label)
        return np.where(
            probabilities >= self.threshold,
            self.positive_label,
            self.negative_label,
        )


def load_predictor(path: str | Path) -> Any:
    """Load a trusted Joblib model artifact.

    Joblib and pickle files can execute code while loading. This function must
    therefore be used only with artifacts produced by the project or another
    explicitly trusted source.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Predictor file not found: {source}")
    return joblib.load(source)