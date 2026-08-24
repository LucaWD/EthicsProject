"""PSyKE CART extraction.

The extractor follows the feature contract of the supplied black box. Feature
availability and temporal validity are evaluated separately by the notebook;
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .evaluation import symbolic_surrogate_metrics


POSITIVE_SYMBOL = "readmitted"
NEGATIVE_SYMBOL = "not_readmitted"


def _safe_identifier(name: str) -> str:
    """Convert a dataframe column into a stable logic-friendly identifier."""

    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "feature"
    if safe[0].isdigit():
        safe = f"feature_{safe}"
    return safe


@dataclass(frozen=True)
class FeatureNameMap:
    """Maintain a reversible mapping between notebook and PSyKE column names."""

    original_to_safe: dict[str, str]
    safe_to_original: dict[str, str]

    @classmethod
    def from_columns(cls, columns: Iterable[str]) -> "FeatureNameMap":
        original_to_safe: dict[str, str] = {}
        safe_to_original: dict[str, str] = {}
        for original in columns:
            original = str(original)
            base = _safe_identifier(original)
            safe = base
            suffix = 2
            while safe in safe_to_original and safe_to_original[safe] != original:
                safe = f"{base}_{suffix}"
                suffix += 1
            original_to_safe[original] = safe
            safe_to_original[safe] = original
        if len(original_to_safe) != len(safe_to_original):
            raise RuntimeError("Feature-name sanitization produced a non-reversible map.")
        return cls(original_to_safe=original_to_safe, safe_to_original=safe_to_original)

    @property
    def original_features(self) -> tuple[str, ...]:
        return tuple(self.original_to_safe)

    @property
    def safe_features(self) -> tuple[str, ...]:
        return tuple(self.safe_to_original)

    def to_safe(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.original_features if column not in frame]
        if missing:
            raise KeyError(f"Cannot build PSyKE view; missing original features: {missing}")
        return frame.loc[:, self.original_features].rename(columns=self.original_to_safe)

    def to_original(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.safe_features if column not in frame]
        if missing:
            raise KeyError(f"Cannot restore model view; missing safe features: {missing}")
        restored = frame.loc[:, self.safe_features].rename(columns=self.safe_to_original)
        return restored.loc[:, self.original_features]


class StringLabelPredictor:
    """Expose a numeric binary classifier as a string-label oracle for PSyKE.

    PSyKE 1.0.4 selects ``DecisionTreeClassifier`` only when the oracle output
    is made of strings. Returning numeric 0/1 would make its CART extractor
    instantiate a regressor instead.
    """

    def __init__(
        self,
        predictor: Any,
        feature_map: FeatureNameMap,
        *,
        positive_label: int = 1,
        negative_label: int = 0,
        positive_symbol: str = POSITIVE_SYMBOL,
        negative_symbol: str = NEGATIVE_SYMBOL,
    ) -> None:
        self.predictor = predictor
        self.feature_map = feature_map
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.positive_symbol = positive_symbol
        self.negative_symbol = negative_symbol
        self.classes_ = np.asarray([negative_symbol, positive_symbol], dtype=object)
        self.feature_names_in_ = np.asarray(feature_map.safe_features, dtype=object)

    def predict(self, safe_frame: pd.DataFrame) -> np.ndarray:
        original_frame = self.feature_map.to_original(safe_frame)
        numeric = np.asarray(self.predictor.predict(original_frame))
        unexpected = set(np.unique(numeric).tolist()) - {
            self.negative_label,
            self.positive_label,
        }
        if unexpected:
            raise ValueError(f"The oracle produced unexpected labels: {unexpected}")
        return np.where(
            numeric == self.positive_label,
            self.positive_symbol,
            self.negative_symbol,
        )


class PrecomputedLabelOracle:
    """Expose stored black-box labels for a restricted feature view.

    This oracle supports surrogate-only ablations. The black box is queried on
    its complete feature matrix first; PSyKE then receives the same row labels
    together with a feature matrix from which one or more explanatory inputs
    have been removed.

    The object is intentionally limited to rows from the original extraction
    pool. It is not a deployable predictive model.
    """

    def __init__(self, X_visible: pd.DataFrame, labels: Iterable[int]) -> None:
        if not isinstance(X_visible, pd.DataFrame) or X_visible.empty:
            raise ValueError("X_visible must be a non-empty pandas DataFrame.")
        if X_visible.columns.has_duplicates:
            raise ValueError("X_visible contains duplicate feature names.")

        numeric_labels = np.asarray(list(labels), dtype=int)
        if numeric_labels.shape != (len(X_visible),):
            raise ValueError("labels must contain one value per visible row.")
        observed = set(np.unique(numeric_labels).tolist())
        if not observed.issubset({0, 1}):
            raise ValueError(f"labels must be binary; found {sorted(observed)}.")

        self.columns = tuple(str(column) for column in X_visible.columns)
        self.classes_ = np.asarray([0, 1])
        self.feature_names_in_ = np.asarray(self.columns, dtype=object)
        row_hashes = pd.util.hash_pandas_object(
            X_visible.loc[:, self.columns],
            index=False,
        ).to_numpy(dtype=np.uint64)

        self._label_by_hash: dict[int, int] = {}
        for row_hash, label in zip(row_hashes, numeric_labels):
            key = int(row_hash)
            value = int(label)
            previous = self._label_by_hash.get(key)
            if previous is not None and previous != value:
                raise ValueError(
                    "Two rows are identical in the restricted feature view but "
                    "receive different black-box labels. A deterministic "
                    "surrogate cannot reproduce both decisions without the "
                    "removed feature."
                )
            self._label_by_hash[key] = value

    def predict(self, X_visible: pd.DataFrame) -> np.ndarray:
        """Return stored labels for known rows in the restricted view."""

        missing_columns = [
            column for column in self.columns if column not in X_visible
        ]
        if missing_columns:
            raise KeyError(
                f"Restricted feature view is missing columns: {missing_columns}"
            )
        row_hashes = pd.util.hash_pandas_object(
            X_visible.loc[:, self.columns],
            index=False,
        ).to_numpy(dtype=np.uint64)
        unknown = [
            int(row_hash)
            for row_hash in row_hashes
            if int(row_hash) not in self._label_by_hash
        ]
        if unknown:
            raise KeyError(
                "The precomputed oracle received rows outside its extraction pool."
            )
        return np.asarray(
            [self._label_by_hash[int(row_hash)] for row_hash in row_hashes],
            dtype=int,
        )


@dataclass
class SymbolicExtractionResult:
    """Artifacts shared by the official PSyKE and diagnostic sklearn paths."""

    backend: str
    extractor: Any
    theory: Any | None
    feature_map: FeatureNameMap
    oracle: StringLabelPredictor
    extraction_sample_size: int
    n_rules: int
    binary_original_features: tuple[str, ...]
    output_name: str = "readmission_prediction"


def _validate_extraction_parameters(
    max_samples: int,
    max_depth: int,
    max_leaves: int | None,
) -> None:
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2.")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")
    if max_leaves is not None and max_leaves < 2:
        raise ValueError("max_leaves must be at least 2 when specified.")


def balanced_oracle_sample(
    X: pd.DataFrame,
    predictor: Any,
    *,
    max_samples: int = 8_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample both black-box decision classes for surrogate training.

    Balancing here concerns the oracle's decisions, not the historical target.
    This lets a compact surrogate observe enough positive and negative regions
    despite the readmission class imbalance.
    """

    if len(X) < 2:
        raise ValueError("At least two rows are required for symbolic extraction.")
    predictions = np.asarray(predictor.predict(X))
    classes = np.unique(predictions)
    if len(classes) < 2:
        raise ValueError(
            "The black-box predictor produced only one class on the extraction pool. "
            "A classification rule set would not be informative."
        )
    if len(X) <= max_samples:
        return X.copy()

    rng = np.random.default_rng(random_state)
    positions_by_class = [np.flatnonzero(predictions == label) for label in classes]
    selected: list[int] = []
    target_per_class = max_samples // len(classes)
    for positions in positions_by_class:
        take = min(len(positions), target_per_class)
        selected.extend(rng.choice(positions, size=take, replace=False).tolist())

    remaining_slots = max_samples - len(selected)
    if remaining_slots:
        available = np.setdiff1d(
            np.arange(len(X)),
            np.asarray(selected, dtype=int),
            assume_unique=False,
        )
        selected.extend(
            rng.choice(available, size=min(remaining_slots, len(available)), replace=False).tolist()
        )
    rng.shuffle(selected)
    return X.iloc[selected].copy()


def _binary_features(frame: pd.DataFrame) -> tuple[str, ...]:
    binary: list[str] = []
    for column in frame.columns:
        values = set(pd.Series(frame[column]).dropna().unique().tolist())
        if values and values.issubset({0, 1, 0.0, 1.0, False, True}):
            binary.append(str(column))
    return tuple(binary)


def extract_psyke_cart(
    predictor: Any,
    X_extraction_pool: pd.DataFrame,
    *,
    max_samples: int = 8_000,
    max_depth: int = 4,
    max_leaves: int | None = 12,
    max_features: int | float | str | None = None,
    simplify: bool = True,
    random_state: int = 42,
) -> SymbolicExtractionResult:
    """Extract a compact classification theory using official PSyKE CART."""

    _validate_extraction_parameters(max_samples, max_depth, max_leaves)
    try:
        from psyke.extraction.cart import Cart
    except ImportError as exc:  # pragma: no cover - exercised in Python 3.11 env
        raise RuntimeError(
            "PSyKE is not installed. Create the documented Python 3.11 environment "
            "and install the project with: pip install -e '.[psyke]'"
        ) from exc

    feature_map = FeatureNameMap.from_columns(X_extraction_pool.columns)
    sample = balanced_oracle_sample(
        X_extraction_pool,
        predictor,
        max_samples=max_samples,
        random_state=random_state,
    )
    safe_sample = feature_map.to_safe(sample)
    oracle = StringLabelPredictor(predictor, feature_map)
    output_name = "readmission_prediction"
    while output_name in safe_sample.columns:
        output_name = f"{output_name}_output"
    extraction_frame = safe_sample.copy()
    # PedagogicalExtractor replaces this placeholder with oracle predictions.
    # Keeping the output last is an explicit PSyKE dataframe requirement.
    extraction_frame[output_name] = NEGATIVE_SYMBOL

    extractor = Cart(
        oracle,
        max_depth=max_depth,
        max_leaves=max_leaves,
        max_features=max_features,
        simplify=simplify,
    )
    theory = extractor.extract(extraction_frame)
    return SymbolicExtractionResult(
        backend="psyke_cart",
        extractor=extractor,
        theory=theory,
        feature_map=feature_map,
        oracle=oracle,
        extraction_sample_size=len(sample),
        n_rules=int(extractor.n_rules),
        binary_original_features=_binary_features(sample),
        output_name=output_name,
    )


def evaluate_symbolic_extraction(
    result: SymbolicExtractionResult,
    X_evaluation: pd.DataFrame,
    y_true: Iterable[int] | None = None,
) -> dict[str, float | int | str]:
    """Measure the rule surrogate against unseen black-box decisions."""

    safe_evaluation = result.feature_map.to_safe(X_evaluation)
    reference = result.oracle.predict(safe_evaluation)
    symbolic = np.asarray(result.extractor.predict(safe_evaluation), dtype=object)
    metrics = symbolic_surrogate_metrics(reference, symbolic, y_true)
    rules = compact_rule_table(result)
    rule_lengths = rules["rule_length"].to_numpy(dtype=float)
    tree = _underlying_tree(result)
    metrics.update(
        {
            "backend": result.backend,
            "n_rules": int(result.n_rules),
            "extraction_sample_size": int(result.extraction_sample_size),
            "n_distinct_features_used": int(len(selected_rule_features(result))),
            "mean_rule_length": float(np.mean(rule_lengths)),
            "median_rule_length": float(np.median(rule_lengths)),
            "max_rule_length": int(np.max(rule_lengths)),
            "surrogate_tree_depth": int(tree.get_depth()),
        }
    )
    return metrics


def _underlying_tree(result: SymbolicExtractionResult) -> Any:
    if result.backend == "psyke_cart":
        try:
            return result.extractor._cart_predictor.predictor
        except AttributeError as exc:  # pragma: no cover - PSyKE version guard
            raise RuntimeError(
                "The installed PSyKE version no longer exposes its CART predictor. "
                "The logical theory is still valid, but compact rendering needs an update."
            ) from exc
    raise ValueError(f"Unsupported symbolic backend: {result.backend}")


def selected_rule_features(result: SymbolicExtractionResult) -> tuple[str, ...]:
    """Return original feature names actually selected by the compact tree."""

    tree = _underlying_tree(result)
    selected_safe = {
        str(tree.feature_names_in_[feature_index])
        for feature_index in tree.tree_.feature
        if feature_index >= 0
    }
    return tuple(
        original
        for original in result.feature_map.original_features
        if result.feature_map.original_to_safe[original] in selected_safe
    )


_ONE_HOT_PREFIX_LABELS: tuple[tuple[str, str], ...] = (
    ("race_", "race"),
    ("age_", "age band"),
    ("admission_type_id_", "admission type code"),
    ("discharge_disposition_id_", "discharge disposition code"),
    ("admission_source_id_", "admission source code"),
    ("payer_code_", "payer code"),
    ("medical_specialty_", "medical specialty"),
    ("diag_1_cat_", "primary diagnosis group"),
    ("diag_2_cat_", "secondary diagnosis group"),
    ("diag_3_cat_", "tertiary diagnosis group"),
)


def _binary_condition(feature: str, operator: str, threshold: float) -> str | None:
    if not np.isclose(threshold, 0.5, atol=0.05):
        return None
    is_present = operator == ">"
    for prefix, label in _ONE_HOT_PREFIX_LABELS:
        if feature.startswith(prefix):
            category = feature[len(prefix) :].replace("_", " ")
            verb = "is" if is_present else "is not"
            return f"{label} {verb} {category}"
    readable = feature.replace("_", " ")
    return f"{readable} is {'true' if is_present else 'false'}"


def _human_condition(
    feature: str,
    operator: str,
    threshold: float,
    binary_features: set[str],
) -> str:
    if feature in binary_features:
        rendered = _binary_condition(feature, operator, threshold)
        if rendered is not None:
            return rendered
    return f"{feature.replace('_', ' ')} {operator} {threshold:.4g}"


def _consolidate_path(
    path: list[tuple[str, str, float]],
) -> list[tuple[str, float | None, float | None]]:
    """Merge repeated tree tests into one lower/upper interval per feature."""

    order: list[str] = []
    bounds: dict[str, dict[str, float | None]] = {}
    for feature, operator, threshold in path:
        if feature not in bounds:
            order.append(feature)
            bounds[feature] = {"lower": None, "upper": None}
        if operator == ">":
            previous = bounds[feature]["lower"]
            bounds[feature]["lower"] = (
                threshold if previous is None else max(previous, threshold)
            )
        elif operator == "<=":
            previous = bounds[feature]["upper"]
            bounds[feature]["upper"] = (
                threshold if previous is None else min(previous, threshold)
            )
        else:
            raise ValueError(f"Unsupported tree operator: {operator}")
    return [
        (feature, bounds[feature]["lower"], bounds[feature]["upper"])
        for feature in order
    ]


def _render_consolidated_path(
    path: list[tuple[str, str, float]],
    binary_features: set[str],
) -> tuple[list[str], list[str]]:
    human: list[str] = []
    machine: list[str] = []
    for feature, lower, upper in _consolidate_path(path):
        if lower is not None and upper is not None:
            readable = feature.replace("_", " ")
            human.append(f"{lower:.4g} < {readable} <= {upper:.4g}")
            machine.append(f"{lower:.10g} < {feature} <= {upper:.10g}")
        elif lower is not None:
            human.append(_human_condition(feature, ">", lower, binary_features))
            machine.append(f"{feature} > {lower:.10g}")
        elif upper is not None:
            human.append(_human_condition(feature, "<=", upper, binary_features))
            machine.append(f"{feature} <= {upper:.10g}")
    return human, machine


def compact_rule_table(result: SymbolicExtractionResult) -> pd.DataFrame:
    """Render each tree leaf as one concise IF/THEN rule."""

    predictor = _underlying_tree(result)
    tree = predictor.tree_
    classes = np.asarray(predictor.classes_, dtype=object)
    binary_features = set(result.binary_original_features)
    records: list[dict[str, Any]] = []

    def visit(node: int, path: list[tuple[str, str, float]]) -> None:
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:  # sklearn uses -1 for both children at a leaf
            values = np.asarray(tree.value[node]).reshape(-1)
            predicted = str(classes[int(np.argmax(values))])
            total_weight = float(values.sum())
            purity = float(values.max() / total_weight) if total_weight else 0.0
            human_conditions, machine_conditions = _render_consolidated_path(
                path,
                binary_features,
            )
            records.append(
                {
                    "rule_id": len(records) + 1,
                    "if": " AND ".join(human_conditions) if human_conditions else "always",
                    "then": predicted,
                    "training_samples": int(tree.n_node_samples[node]),
                    "leaf_purity": purity,
                    "rule_length": len(human_conditions),
                    "tree_path_length": len(path),
                    "machine_condition": " AND ".join(machine_conditions)
                    if machine_conditions
                    else "always",
                }
            )
            return

        safe_feature = str(predictor.feature_names_in_[tree.feature[node]])
        original_feature = result.feature_map.safe_to_original[safe_feature]
        threshold = float(tree.threshold[node])
        visit(left, [*path, (original_feature, "<=", threshold)])
        visit(right, [*path, (original_feature, ">", threshold)])

    visit(0, [])
    return pd.DataFrame.from_records(records)


def save_psyke_theory(
    result: SymbolicExtractionResult,
    path: str | Path,
) -> Path:
    """Save the official PSyKE theory in readable Prolog syntax."""

    if result.backend != "psyke_cart" or result.theory is None:
        raise ValueError("Only an official PSyKE result contains a logical theory.")
    try:
        from psyke.utils.logic import pretty_theory
    except ImportError as exc:  # pragma: no cover - PSyKE environment only
        raise RuntimeError("PSyKE is required to format its logical theory.") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(pretty_theory(result.theory), encoding="utf-8")
    return destination