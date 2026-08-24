"""Reusable components for safety and symbolic model analysis."""

from .evaluation import (
    binary_classification_metrics,
    class_conditional_fidelity_table,
    confusion_table,
    symbolic_surrogate_metrics,
    symbolic_task_metrics_on_recognized,
)
from .modeling import (
    ThresholdedClassifier,
    load_predictor,
    positive_class_probability,
)
from .safety import (
    DeceasedStatusDetector,
    SafetyWrapper,
    assert_safety_invariants,
    safety_impact_summary,
)
from .symbolic import (
    PrecomputedLabelOracle,
    SymbolicExtractionResult,
    balanced_oracle_sample,
    compact_rule_table,
    evaluate_symbolic_extraction,
    extract_psyke_cart,
    save_psyke_theory,
    selected_rule_features,
)

__all__ = [
    "DeceasedStatusDetector",
    "PrecomputedLabelOracle",
    "SafetyWrapper",
    "SymbolicExtractionResult",
    "ThresholdedClassifier",
    "assert_safety_invariants",
    "balanced_oracle_sample",
    "binary_classification_metrics",
    "class_conditional_fidelity_table",
    "compact_rule_table",
    "confusion_table",
    "evaluate_symbolic_extraction",
    "extract_psyke_cart",
    "load_predictor",
    "positive_class_probability",
    "safety_impact_summary",
    "save_psyke_theory",
    "selected_rule_features",
    "symbolic_surrogate_metrics",
    "symbolic_task_metrics_on_recognized",
]