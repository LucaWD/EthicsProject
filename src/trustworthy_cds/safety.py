"""A model-independent safety guard for non-eligible deceased patients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


SAFETY_REASON = "DECEASED_PATIENT_NOT_ELIGIBLE_FOR_READMISSION"


def _as_binary_series(values: pd.Series, column: str) -> pd.Series:
    """Validate a status signal and return it as a boolean series."""

    if values.isna().any():
        raise ValueError(f"Safety status column '{column}' contains missing values.")
    numeric = pd.to_numeric(values, errors="raise")
    observed = set(numeric.unique().tolist())
    if not observed.issubset({0, 1, False, True}):
        raise ValueError(
            f"Safety status column '{column}' must be binary; found {sorted(observed)}."
        )
    return numeric.astype(bool)


@dataclass(frozen=True)
class DeceasedStatusDetector:
    """Detect deceased rows from an explicit signal or historical dummy codes.

    At deployment time, ``is_deceased`` should come from the live clinical
    workflow. The discharge-disposition one-hot columns are supported only to
    verify the rule retrospectively on the available historical dataset.
    """

    deceased_columns: tuple[str, ...] = (
        "discharge_disposition_id_11",
        "discharge_disposition_id_19",
        "discharge_disposition_id_20",
        "discharge_disposition_id_21",
    )
    explicit_status_column: str = "is_deceased"

    def detect(self, context: pd.DataFrame) -> pd.Series:
        if not isinstance(context, pd.DataFrame):
            raise TypeError("Safety context must be a pandas DataFrame.")
        signals: list[pd.Series] = []
        if self.explicit_status_column in context:
            signals.append(
                _as_binary_series(
                    context[self.explicit_status_column],
                    self.explicit_status_column,
                )
            )
        for column in self.deceased_columns:
            if column in context:
                signals.append(_as_binary_series(context[column], column))
        if not signals:
            expected = [self.explicit_status_column, *self.deceased_columns]
            raise KeyError(
                "No deceased-status signal is available in the safety context. "
                f"Expected one of: {expected}"
            )

        detected = pd.concat(signals, axis=1).any(axis=1)
        detected.name = "is_deceased"
        return detected


class SafetyWrapper:
    """Override a classifier only when a hard safety rule is triggered.

    The wrapped predictor receives model features only. Safety columns are read
    from the separate context dataframe, preventing post-outcome information
    from becoming a predictive shortcut.
    """

    def __init__(
        self,
        predictor: Any,
        model_features: Sequence[str],
        detector: DeceasedStatusDetector | None = None,
        *,
        positive_label: int = 1,
        negative_label: int = 0,
    ) -> None:
        if not hasattr(predictor, "predict") or not hasattr(predictor, "predict_proba"):
            raise TypeError("The predictor must expose predict() and predict_proba().")
        if not hasattr(predictor, "classes_"):
            raise TypeError("The predictor must expose classes_.")
        if not model_features:
            raise ValueError("model_features cannot be empty.")
        if len(set(model_features)) != len(model_features):
            raise ValueError("model_features contains duplicate names.")

        self.predictor = predictor
        self.model_features = tuple(model_features)
        self.detector = detector or DeceasedStatusDetector()
        self.positive_label = positive_label
        self.negative_label = negative_label

        classes = np.asarray(predictor.classes_)
        if set(classes.tolist()) != {negative_label, positive_label}:
            raise ValueError(
                "The predictor must contain exactly the configured positive and negative labels."
            )

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.predictor.classes_)

    def _model_input(self, context: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.model_features if column not in context]
        if missing:
            raise KeyError(f"Safety context is missing model features: {missing}")
        return context.loc[:, self.model_features]

    def _class_index(self, label: int) -> int:
        matching = np.flatnonzero(self.classes_ == label)
        if len(matching) != 1:
            raise ValueError(f"Label {label!r} is not uniquely present in classes_.")
        return int(matching[0])

    def predict_proba(self, context: pd.DataFrame) -> np.ndarray:
        model_input = self._model_input(context)
        probabilities = np.asarray(self.predictor.predict_proba(model_input), dtype=float)
        if probabilities.shape != (len(context), len(self.classes_)):
            raise ValueError("The predictor returned an invalid probability matrix shape.")
        if not np.isfinite(probabilities).all():
            raise ValueError("The predictor returned non-finite probabilities.")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("The predictor returned probabilities outside [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each predictor probability row must sum to one.")

        deceased = self.detector.detect(context).to_numpy()
        safe_probabilities = probabilities.copy()
        if deceased.any():
            safe_probabilities[deceased, :] = 0.0
            safe_probabilities[deceased, self._class_index(self.negative_label)] = 1.0
        return safe_probabilities

    def predict(self, context: pd.DataFrame) -> np.ndarray:
        model_input = self._model_input(context)
        base_predictions = np.asarray(self.predictor.predict(model_input)).copy()
        if base_predictions.shape != (len(context),):
            raise ValueError("The predictor returned an invalid prediction vector shape.")
        deceased = self.detector.detect(context).to_numpy()
        base_predictions[deceased] = self.negative_label
        return base_predictions

    def predict_with_audit(self, context: pd.DataFrame) -> pd.DataFrame:
        """Return final decisions together with the evidence for each override."""

        model_input = self._model_input(context)
        deceased = self.detector.detect(context)
        base_predictions = np.asarray(self.predictor.predict(model_input))
        base_probabilities = np.asarray(self.predictor.predict_proba(model_input), dtype=float)
        positive_index = self._class_index(self.positive_label)

        final_predictions = self.predict(context)
        final_probabilities = self.predict_proba(context)[:, positive_index]
        overridden = deceased.to_numpy()
        return pd.DataFrame(
            {
                "base_prediction": base_predictions,
                "base_readmission_probability": base_probabilities[:, positive_index],
                "is_deceased": deceased.to_numpy(),
                "safety_rule_triggered": overridden,
                "final_prediction": final_predictions,
                "final_readmission_probability": final_probabilities,
                "safety_reason": np.where(overridden, SAFETY_REASON, ""),
            },
            index=context.index,
        )


def assert_safety_invariants(
    wrapper: SafetyWrapper,
    context: pd.DataFrame,
) -> dict[str, int | bool]:
    """Fail loudly unless deceased and non-deceased behavior is correct."""

    audit = wrapper.predict_with_audit(context)
    deceased = audit["is_deceased"].to_numpy(dtype=bool)
    if deceased.any():
        if not (audit.loc[deceased, "final_prediction"] == wrapper.negative_label).all():
            raise AssertionError("A deceased row retained a positive final prediction.")
        if not np.allclose(
            audit.loc[deceased, "final_readmission_probability"].to_numpy(),
            0.0,
        ):
            raise AssertionError("A deceased row retained non-zero readmission probability.")

    alive = ~deceased
    if alive.any():
        if not np.array_equal(
            audit.loc[alive, "base_prediction"].to_numpy(),
            audit.loc[alive, "final_prediction"].to_numpy(),
        ):
            raise AssertionError("The wrapper changed a non-deceased prediction.")
        if not np.allclose(
            audit.loc[alive, "base_readmission_probability"].to_numpy(),
            audit.loc[alive, "final_readmission_probability"].to_numpy(),
        ):
            raise AssertionError("The wrapper changed a non-deceased probability.")

    return {
        "n_rows_checked": int(len(audit)),
        "n_deceased_rows": int(deceased.sum()),
        "n_non_deceased_rows": int(alive.sum()),
        "all_invariants_passed": True,
    }


def safety_impact_summary(audit: pd.DataFrame) -> pd.Series:
    """Summarize exactly what a completed safety audit changed.

    The summary distinguishes hard-label changes from probability changes and
    verifies that the effect is confined to deceased encounters. It is a
    reporting helper; it does not call the predictor or modify the audit.
    """

    required_columns = (
        "is_deceased",
        "base_prediction",
        "final_prediction",
        "base_readmission_probability",
        "final_readmission_probability",
    )
    missing = [column for column in required_columns if column not in audit]
    if missing:
        raise KeyError(f"Safety audit is missing required columns: {missing}")

    deceased = audit["is_deceased"].astype(bool)
    prediction_changed = audit["base_prediction"] != audit["final_prediction"]
    probability_changed = pd.Series(
        ~np.isclose(
            audit["base_readmission_probability"],
            audit["final_readmission_probability"],
        ),
        index=audit.index,
    )

    deceased_probabilities = audit.loc[
        deceased,
        ["base_readmission_probability", "final_readmission_probability"],
    ]
    maximum_base = (
        float(deceased_probabilities["base_readmission_probability"].max())
        if len(deceased_probabilities)
        else float("nan")
    )
    maximum_final = (
        float(deceased_probabilities["final_readmission_probability"].max())
        if len(deceased_probabilities)
        else float("nan")
    )

    return pd.Series(
        {
            "deceased rows": int(deceased.sum()),
            "deceased predicted readmitted before wrapper": int(
                (deceased & (audit["base_prediction"] == 1)).sum()
            ),
            "predictions actually changed": int(
                (deceased & prediction_changed).sum()
            ),
            "probabilities actually changed": int(
                (deceased & probability_changed).sum()
            ),
            "non-deceased predictions changed": int(
                ((~deceased) & prediction_changed).sum()
            ),
            "non-deceased probabilities changed": int(
                ((~deceased) & probability_changed).sum()
            ),
            "maximum base probability among deceased": maximum_base,
            "maximum final probability among deceased": maximum_final,
        },
        name="value",
    )