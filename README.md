# EthicsProject

Trustworthy clinical decision support for the prediction of 30-day hospital readmission using the Diabetes 130-US Hospitals dataset.

The project is organized into four analysis stages: data preprocessing, predictive modelling and robustness, fairness assessment and mitigation, and symbolic explainability with safety verification. The preprocessing pipeline produces a shared model representation containing 148 features, used by the Decision Tree, XGBoost, and Multilayer Perceptron models.

## Repository structure

```text
EthicsProject/
├── 00_preprocessing.ipynb
├── 01_predictive_modeling_and_robustness.ipynb
├── 02_fairness_and_bias_mitigation.ipynb
├── 03_symbolic_explainability_and_safety.ipynb
├── data/
│   ├── diabetic_data.csv
│   ├── IDS_mapping.csv
│   ├── processed/
│   └── standard_scaler.joblib
├── models/
├── reports/
│   └── safety_symbolic_model_analysis/
├── src/
│   ├── robustness/
│   └── trustworthy_cds/
└── theories/
    └── safety_symbolic_model_analysis/
```

- `data/` contains the original dataset, identifier mappings, processed data, train/test partitions, and the fitted scaler.
- `models/` contains the trained classifiers, selected hyperparameters, and decision thresholds.
- `src/robustness/` provides helper functions for the robustness experiments.
- `src/trustworthy_cds/` contains the shared evaluation, modelling, safety, and symbolic-analysis utilities.
- `reports/` stores the exported symbolic and safety results in CSV and JSON format.
- `theories/` contains the extracted executable Prolog theories for the three models.

## Notebooks

| Notebook | Content |
|---|---|
| `00_preprocessing.ipynb` | Data cleaning, feature encoding, diagnosis and specialty grouping, numerical transformations, train/test split, and construction of the 148-feature representation. |
| `01_predictive_modeling_and_robustness.ipynb` | Training and evaluation of Decision Tree, XGBoost, and MLP models, followed by data-poisoning, model-poisoning, evasion, and defence experiments. |
| `02_fairness_and_bias_mitigation.ipynb` | Pre-model and post-model fairness audits across demographic groups, with reweighing, group-specific thresholding, and intersectional analysis. |
| `03_symbolic_explainability_and_safety.ipynb` | Extraction and evaluation of compact PSyKE/CART rules, analysis of discharge-disposition features, and verification of the safety constraint. |


