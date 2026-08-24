import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42

# Score-based tabular attack parameters.
TABULAR_EPSILON = 0.50
TABULAR_STEP_FRACTION = 0.10
TABULAR_MAX_ITER = 5

# Feature squeezing and randomized smoothing parameters.
SQUEEZE_DECIMALS = 1
SMOOTHING_NOISE_STD = 0.05
SMOOTHING_SAMPLES = 25

def identify_perturbable_columns(X):
    """
    Return numeric columns that are not binary or one-hot encoded variables.
    """
    cols = []
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            continue
        values = pd.Series(X[column]).dropna().unique()
        if len(values) <= 2:
            try:
                numeric_values = np.round(values.astype(float), 8)
                if set(numeric_values).issubset({0.0, 1.0}):
                    continue
            except Exception:
                pass

        cols.append(column)
    return cols

def clean_correct_subset(model, threshold, X, y):
    """
    Keep only samples that are classified correctly before the attack.
    """
    X = X.copy()
    y = np.asarray(y).astype(int)

    probs = model.predict_proba(pd.DataFrame(X, columns=X.columns))[:, 1]
    predictions = (np.asarray(probs) >= threshold).astype(int)
    mask = predictions == y

    return (X.loc[mask].copy(), y[mask].copy(), probs[mask].copy())

def enforce_tabular_constraints(X_adv, X_original, PERTURBABLE_IDX, TRAIN_MIN, TRAIN_MAX):
    """
    Clip perturbable continuous variables to their training range and
    restore all frozen variables from the original sample.
    """
    if isinstance(X_adv, pd.DataFrame):
        X_adv = X_adv.to_numpy(dtype=np.float32, copy=True)
    else:
        X_adv = np.asarray(X_adv, dtype=np.float32).copy()

    if isinstance(X_original, pd.DataFrame):
        X_original = X_original.to_numpy(dtype=np.float32, copy=True)
    else:
        X_original = np.asarray(X_original, dtype=np.float32).copy()

    X_adv[:, PERTURBABLE_IDX] = np.clip(X_adv[:, PERTURBABLE_IDX],TRAIN_MIN,TRAIN_MAX)

    frozen_idx = np.setdiff1d(np.arange(X_adv.shape[1]),PERTURBABLE_IDX,)
    X_adv[:, frozen_idx] = X_original[:, frozen_idx]

    return X_adv

def sample_binary_log_loss(y_true, y_prob, eps=1e-7):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), eps, 1.0-eps)

    return -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def tabular_score_attack(
    model,
    X,
    y,
    PERTURBABLE_IDX,
    TRAIN_STD,
    TRAIN_MIN,
    TRAIN_MAX,
    epsilon=TABULAR_EPSILON,
    step_fraction=TABULAR_STEP_FRACTION,
    max_iter=TABULAR_MAX_ITER,
    seed=SEED,
):
    """
    Score-based coordinate attack designed specifically for tabular data.

    For each feature, two candidate perturbations are evaluated:
        x_j + step
        x_j - step

    The perturbation increasing the loss of the true class is retained.
    """
    rng = np.random.default_rng(seed)

    X_original = X.to_numpy(dtype=np.float32, copy=True)
    X_adv = X_original.copy()
    y = np.asarray(y).astype(int)

    current_df = pd.DataFrame(X_adv, columns=X.columns)
    current_probability = model.predict_proba(current_df)[:, 1]
    current_loss = sample_binary_log_loss(y, current_probability)

    std_by_feature = {
        feature_idx: float(TRAIN_STD[i])
        for i, feature_idx in enumerate(PERTURBABLE_IDX)
    }

    for _ in range(max_iter):
        improved = False

        feature_order = PERTURBABLE_IDX.copy()
        rng.shuffle(feature_order)

        for feature_idx in feature_order:
            step = (epsilon * step_fraction * std_by_feature[feature_idx])

            X_plus = X_adv.copy()
            X_minus = X_adv.copy()

            X_plus[:, feature_idx] += step
            X_minus[:, feature_idx] -= step

            X_plus = enforce_tabular_constraints(X_plus,X_original,PERTURBABLE_IDX,TRAIN_MIN,TRAIN_MAX)
            X_minus = enforce_tabular_constraints(X_minus,X_original,PERTURBABLE_IDX,TRAIN_MIN,TRAIN_MAX)

            plus_probability = model.predict_proba(pd.DataFrame(X_plus, columns=X.columns))[:, 1]
            minus_probability = model.predict_proba(pd.DataFrame(X_minus, columns=X.columns))[:, 1]

            plus_loss = sample_binary_log_loss(y,plus_probability)
            minus_loss = sample_binary_log_loss(y,minus_probability)

            take_plus = ((plus_loss > current_loss) & (plus_loss >= minus_loss))
            take_minus = ((minus_loss > current_loss) & (minus_loss > plus_loss))

            if take_plus.any():
                X_adv[take_plus, feature_idx] = (X_plus[take_plus, feature_idx])
                current_loss[take_plus] = plus_loss[take_plus]
                improved = True

            if take_minus.any():
                X_adv[take_minus, feature_idx] = (X_minus[take_minus, feature_idx])
                current_loss[take_minus] = minus_loss[take_minus]
                improved = True

        if not improved:
            break

    X_adv = enforce_tabular_constraints(X_adv,X_original,PERTURBABLE_IDX,TRAIN_MIN,TRAIN_MAX)

    return pd.DataFrame(X_adv, columns=X.columns)

def make_art_estimator(model_name, model, X):
    """
    Wrap a sklearn model using the ART estimator interface.
    """
    from art.estimators.classification.scikitlearn import (
            ScikitlearnClassifier,
            ScikitlearnDecisionTreeClassifier,
        )
    from art.estimators.classification.xgboost import XGBoostClassifier
    
    feature_min = X.min(axis=0).astype(np.float32).to_numpy()
    feature_max = X.max(axis=0).astype(np.float32).to_numpy()

    invalid_mask = feature_min >= feature_max

    if invalid_mask.any():
        feature_max = feature_max.copy()
        feature_max[invalid_mask] = feature_min[invalid_mask] + 1e-6

    clip_values = (feature_min, feature_max)

    if model_name == "Decision Tree":
        return ScikitlearnDecisionTreeClassifier(
            model=model,
            clip_values=clip_values,
        )

    if model_name == "XGBoost":
        return XGBoostClassifier(
            model=model,
            clip_values=clip_values,
            nb_classes=2,
            nb_features=X.shape[1]
        )

    if model_name == "MLP":
        return ScikitlearnClassifier(
            model=model,
            clip_values=clip_values,
            use_logits=False,
        )

def labels_to_one_hot(y):
    """
    Convert binary labels into ART's one-hot representation.
    """
    y = np.asarray(y).astype(int).ravel()
    result = np.zeros((len(y), 2), dtype=np.float32)
    result[np.arange(len(y)), y] = 1.0
    return result


def run_art_attack(
    model_name,
    model,
    attack_name,
    X_train,
    X_attack,
    y_attack,
    PERTURBABLE_IDX,
    TRAIN_MIN,
    TRAIN_MAX
):
    """
    Generate an adversarial dataset with an ART attack.
    """
    from art.attacks.evasion import (
            BoundaryAttack,
            DecisionTreeAttack,
            HopSkipJump
    )
    
    estimator = make_art_estimator(model_name,model,X_train)

    x_np = X_attack.to_numpy(dtype=np.float32, copy=True)
    y_oh = labels_to_one_hot(y_attack)

    try:
        # ============================================================
        # Decision-tree-specific attack
        # ============================================================
        if attack_name == "DecisionTreeAttack":
            if model_name != "Decision Tree":
                return None
            attack = DecisionTreeAttack(
                classifier=estimator,
                offset=0.001,
                verbose=False,
            )
        # ============================================================
        # Decision / black-box attacks
        # ============================================================
        elif attack_name == "HopSkipJump":
            attack = HopSkipJump(
                classifier=estimator,
                targeted=False,
                norm=2,
                max_iter=10,
                max_eval=200,
                init_eval=20,
                init_size=20,
                batch_size=16,
                verbose=False,
            )
        elif attack_name == "BoundaryAttack":
            attack = BoundaryAttack(
                estimator=estimator,
                targeted=False,
                delta=0.1,
                epsilon=0.01,
                max_iter=50,
                num_trial=10,
                sample_size=10,
                init_size=100,
                batch_size=16,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        X_adv = attack.generate(x=x_np,y=y_oh)
        X_adv = enforce_tabular_constraints(X_adv,x_np,PERTURBABLE_IDX,TRAIN_MIN,TRAIN_MAX)

        return pd.DataFrame(X_adv, columns=X_attack.columns)

    except Exception as exc:
        print(f"[{model_name} / {attack_name}] SKIPPED: {type(exc).__name__}: {exc}")
        return None

def attack_statistics(model, X_clean, X_adv, y, threshold, PERTURBABLE_IDX):
    """
    Compute attack success and perturbation statistics.
    """
    y = np.asarray(y).astype(int)
    clean_probs = model.predict_proba(X_clean)[:, 1]
    adv_probs = model.predict_proba(X_adv)[:, 1]

    clean_pred = (np.asarray(clean_probs) >= threshold).astype(int)
    adv_pred =(np.asarray(adv_probs) >= threshold).astype(int)

    correct_before = clean_pred == y
    successful = correct_before & (adv_pred != y)
    positive_before = correct_before & (y == 1)
    successful_positive = successful & (y == 1)

    delta = (
        X_adv.to_numpy(dtype=np.float32, copy=True)[:, PERTURBABLE_IDX]
        - X_clean.to_numpy(dtype=np.float32, copy=True)[:, PERTURBABLE_IDX]
    )

    n_correct = int(correct_before.sum())
    n_positive = int(positive_before.sum())

    return {
        "Clean Correct N": n_correct,
        "Clean Correct Positives": n_positive,
        "Successful Attacks": int(successful.sum()),
        "Successful Positive Attacks": int(successful_positive.sum()),
        "Attack Success Rate": (float(successful.sum() / n_correct) if n_correct else np.nan),
        "Positive Attack Success": (float(successful_positive.sum() / n_positive) if n_positive else np.nan),
        "Mean Probability Change": np.mean(adv_probs - clean_probs),
        "Mean |Probability Change|": np.mean(
            np.abs(adv_probs - clean_probs)
        ),
        "Mean L2": np.mean(np.linalg.norm(delta, axis=1)),
        "Mean Linf": np.mean(np.max(np.abs(delta), axis=1)),
        "Mean Changed Features": np.mean(
            np.sum(np.abs(delta) > 1e-8, axis=1)
        ),
    }

def evaluate_probability_output(y_true, y_prob, threshold):
    """
    Evaluate classification metrics from positive-class probabilities.

    ROC-AUC and PR-AUC are returned as NaN when they are not defined
    (for example, when a rejection defense leaves only one class).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return {
            "Recall": np.nan,
            "Precision": np.nan,
            "F1": np.nan,
            "F2": np.nan,
            "ROC-AUC": np.nan,
            "PR-AUC": np.nan,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "TP": 0,
        }

    preds = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        preds,
        labels=[0, 1],
    ).ravel()

    has_both_classes = np.unique(y_true).size == 2
    roc_auc = (
        roc_auc_score(y_true, y_prob)
        if has_both_classes
        else np.nan
    )
    pr_auc = (
        average_precision_score(y_true, y_prob)
        if has_both_classes
        else np.nan
    )

    return {
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "F2": fbeta_score(y_true, preds, beta=2.0, zero_division=0),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def feature_squeeze(X,PERTURBABLE_COLS,decimals=SQUEEZE_DECIMALS):
    """
    Round perturbable continuous variables.
    """
    X_squeezed = X.copy()
    cast_dict = {col: np.float32 for col in PERTURBABLE_COLS}
    X_squeezed = X_squeezed.astype(cast_dict)
    X_squeezed[PERTURBABLE_COLS] = X_squeezed[PERTURBABLE_COLS].round(decimals)
    return X_squeezed

def evaluate_outlier_rejection(
    model,
    threshold,
    X_input,
    y,
    outlier_detector,
    PERTURBABLE_COLS
):
    """
    Reject samples considered anomalous.
    """
    flags = (outlier_detector.predict(X_input[PERTURBABLE_COLS].astype(float)) == -1)

    accepted = ~flags

    probs = model.predict_proba(X_input)[:, 1]
       
    predictions = (np.asarray(probs) >= threshold).astype(int)

    if accepted.any():
        y_acc = np.asarray(y)[accepted]
        pred_acc = predictions[accepted]

        conditional_recall = recall_score(
            y_acc, pred_acc, zero_division=0
        )
        conditional_precision = precision_score(
            y_acc, pred_acc, zero_division=0
        )
        conditional_f2 = fbeta_score(
            y_acc, pred_acc, beta=2.0, zero_division=0
        )

    else:
        conditional_recall = np.nan
        conditional_precision = np.nan
        conditional_f2 = np.nan

    return {
        "Rejection Rate": flags.mean(),
        "Coverage": accepted.mean(),
        "Conditional Recall": conditional_recall,
        "Conditional Precision": conditional_precision,
        "Conditional F2": conditional_f2,
    }

def randomized_smoothing_predict_proba(
    model,
    X,
    PERTURBABLE_IDX,
    TRAIN_STD,
    TRAIN_MIN,
    TRAIN_MAX,
    noise_std=SMOOTHING_NOISE_STD,
    n_samples=SMOOTHING_SAMPLES,
    seed=SEED
):
    """
    Inject noise only into continuous perturbable features.
    """
    if isinstance(X, pd.DataFrame):
        X_base =  X.to_numpy(dtype=np.float32, copy=True)
    else:
        X_base = np.asarray(X, dtype=np.float32).copy()

    if len(PERTURBABLE_IDX) == 0:
        base_prob = model.predict_proba(X)[:, 1]
        return np.asarray(base_prob, dtype=float)

    rng = np.random.default_rng(seed)
    probability_samples = []
    
    for _ in range(n_samples):
        X_noisy = X_base.copy()

        noise = rng.normal(
            0.0,
            noise_std * TRAIN_STD,
            size=(len(X_noisy),len(PERTURBABLE_IDX),)
        )

        X_noisy[:, PERTURBABLE_IDX] += noise
        X_noisy = enforce_tabular_constraints(
            X_noisy,
            X_base,
            PERTURBABLE_IDX,
            TRAIN_MIN,
            TRAIN_MAX
        )

        probability_samples.append(
            model.predict_proba(
                pd.DataFrame(
                    X_noisy,
                    columns=X.columns,
                    index=X.index,
                )
            )[:, 1]
        )

    return np.mean(np.vstack(probability_samples), axis=0)

def adversarial_train(
    model_name,
    model,
    threshold,
    X_train,
    y_train,
    PERTURBABLE_IDX,
    TRAIN_STD,
    TRAIN_MIN,
    TRAIN_MAX,
    AT_TRAIN_SAMPLE_SIZE,
    seed,
    return_training_data=False,
):
    """
    Train a fresh model on clean data plus score-based adversarial examples.

    The clean subset is sampled randomly rather than taking the first rows,
    avoiding dependence on the original row ordering.

    When ``return_training_data`` is True, the augmented training set is
    returned as well. This is useful for applying data-poisoning experiments
    to the *actual* training set used by the defense.
    """
    rng = np.random.default_rng(seed)
    n = min(AT_TRAIN_SAMPLE_SIZE, len(X_train))
    subset_idx = rng.choice(len(X_train), size=n, replace=False)

    X_subset = X_train.iloc[subset_idx].copy()
    y_subset = np.asarray(y_train)[subset_idx].astype(int)

    X_clean, y_clean, _ = clean_correct_subset(
        model,
        threshold,
        X_subset,
        y_subset,
    )

    if len(X_clean) > 0:
        X_adv = tabular_score_attack(
            model,
            X_clean,
            y_clean,
            PERTURBABLE_IDX=PERTURBABLE_IDX,
            TRAIN_STD=TRAIN_STD,
            TRAIN_MIN=TRAIN_MIN,
            TRAIN_MAX=TRAIN_MAX,
            seed=seed,
        )
        X_aug = pd.concat(
            [
                X_train.reset_index(drop=True),
                X_adv.reset_index(drop=True),
            ],
            ignore_index=True,
        ).astype(X_train.dtypes)
        y_aug = np.concatenate(
            [
                np.asarray(y_train).astype(int),
                y_clean,
            ]
        )
    else:
        X_aug = X_train.reset_index(drop=True).copy()
        y_aug = np.asarray(y_train).astype(int).copy()

    robust_model = clone(model)
    robust_model.fit(X_aug, y_aug)

    if return_training_data:
        return robust_model, X_aug, y_aug

    return robust_model


def build_ensemble_adversarial_dataset(target_name,X_train,y_train,best_models,optimal_thresholds,PERTURBABLE_IDX,TRAIN_STD,TRAIN_MIN,TRAIN_MAX,AT_TRAIN_SAMPLE_SIZE,seed):
    """
    Generate adversarial samples using the other models and
    augment the clean training set.

    This tests whether exposure to diverse attack/model pairs
    improves robustness beyond single-model adversarial training.
    """
    rng = np.random.default_rng(seed)
    n = min(AT_TRAIN_SAMPLE_SIZE, len(X_train))
    subset_idx = rng.choice(len(X_train), size=n, replace=False)

    X_subset = X_train.iloc[subset_idx].copy()
    y_subset = np.asarray(y_train)[subset_idx].astype(int)

    adversarial_parts = []
    label_parts = []
    for donor_name, donor_model in best_models.items():
        if donor_name == target_name:
            continue

        threshold = float(optimal_thresholds[donor_name])

        X_clean, y_clean, _ = clean_correct_subset(
            donor_model,
            threshold,
            X_subset,
            y_subset,
        )

        if len(X_clean) == 0:
            continue

        X_adv = tabular_score_attack(
            donor_model,
            X_clean,
            y_clean,
            PERTURBABLE_IDX=PERTURBABLE_IDX,
            TRAIN_STD=TRAIN_STD,
            TRAIN_MIN=TRAIN_MIN,
            TRAIN_MAX=TRAIN_MAX,
            seed=SEED + len(adversarial_parts)
        )

        adversarial_parts.append(X_adv.reset_index(drop=True))
        label_parts.append(y_clean)

    X_parts = [X_train.reset_index(drop=True), *adversarial_parts]
    X_aug = pd.concat(X_parts, ignore_index=True).astype(X_train.dtypes)
    y_parts = [np.asarray(y_train).astype(int), *label_parts]
    y_aug = np.concatenate(y_parts)

    return X_aug, y_aug

def poison_training_labels(
    y,
    fraction,
    seed=SEED,
    strategy="random",
):
    """
    Create a poisoned training-label vector.

    ``strategy='random'`` flips the requested fraction of all labels.
    ``strategy='positive_to_negative'`` flips the requested fraction of
    positive labels only, directly targeting recall.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1.")

    y_poisoned = np.asarray(y).astype(int).copy()
    rng = np.random.default_rng(seed)

    if strategy == "random":
        candidates = np.arange(len(y_poisoned))
    elif strategy == "positive_to_negative":
        candidates = np.flatnonzero(y_poisoned == 1)
    else:
        raise ValueError(f"Unknown poisoning strategy: {strategy}")

    n_poison = int(round(fraction * len(candidates)))

    if n_poison == 0:
        return y_poisoned, np.array([], dtype=int)

    poisoned_idx = rng.choice(
        candidates,
        size=min(n_poison, len(candidates)),
        replace=False,
    )
    y_poisoned[poisoned_idx] = 1 - y_poisoned[poisoned_idx]

    return y_poisoned, poisoned_idx


def train_with_data_poisoning(
    model,
    X_train,
    y_train,
    fraction,
    strategy="random",
    seed=SEED,
):
    """
    Clone and retrain a model on a label-poisoned training set.
    """
    y_poisoned, poisoned_idx = poison_training_labels(
        y_train,
        fraction=fraction,
        seed=seed,
        strategy=strategy,
    )

    poisoned_model = clone(model)
    poisoned_model.fit(X_train, y_poisoned)

    return poisoned_model, y_poisoned, poisoned_idx

def poison_model_artifact(
    model,
    model_name,
    severity,
):
    """
    Simulate model poisoning by maliciously corrupting learned model internals.

    This is a model-artifact tampering experiment, not a claim that every
    distributed-training poisoning attack has this exact implementation.

    The attack is deliberately directed toward under-predicting the positive
    class:
      * Decision Tree: reduce positive leaf mass.
      * MLP: decrease the final output bias.
      * XGBoost: reduce the base score.
    """
    if not 0.0 < severity <= 1.0:
        raise ValueError("severity must be in (0, 1].")

    poisoned = deepcopy(model)

    if model_name == "Decision Tree":
        values = poisoned.tree_.value
        values[:, :, 1] *= max(1.0 - severity, 0.0)

    elif model_name == "MLP":
        if not hasattr(poisoned, "intercepts_") or not poisoned.intercepts_:
            raise ValueError("The MLP must be fitted before model poisoning.")

        # The final intercept is a direct learned parameter controlling the
        # positive-class output in the binary MLP.
        poisoned.intercepts_[-1] = (
            poisoned.intercepts_[-1] - float(severity)
        )

    elif model_name == "XGBoost":
        # XGBoost uses base_score as the model's prior prediction level.
        # Lowering it systematically biases predictions toward class 0.
        base_score = poisoned.get_params().get("base_score", 0.5)
        try:
            base_value = float(np.asarray(base_score).reshape(-1)[0])
        except (TypeError, ValueError):
            base_value = 0.5

        poisoned.set_params(
            base_score=max(1e-4, base_value * (1.0 - severity))
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return poisoned

def probability_shift_statistics(
    y,
    clean_probabilities,
    attacked_probabilities,
    threshold,
    X_clean=None,
    X_attacked=None,
    PERTURBABLE_IDX=None,
):
    """
    Measure how an attack changes predictions relative to an intact reference.

    For evasion attacks, paired clean/attacked inputs may be supplied to
    compute L2/Linf perturbation statistics. For data/model poisoning, the
    input does not change, so those norm fields are intentionally NaN.
    """
    y = np.asarray(y).astype(int)
    clean_probabilities = np.asarray(clean_probabilities, dtype=float)
    attacked_probabilities = np.asarray(attacked_probabilities, dtype=float)

    clean_pred = (clean_probabilities >= threshold).astype(int)
    attacked_pred = (attacked_probabilities >= threshold).astype(int)

    correct_before = clean_pred == y
    successful = correct_before & (attacked_pred != y)
    positive_before = correct_before & (y == 1)
    successful_positive = successful & (y == 1)

    n_correct = int(correct_before.sum())
    n_positive = int(positive_before.sum())

    mean_l2 = np.nan
    mean_linf = np.nan
    mean_changed = np.nan

    if (
        X_clean is not None
        and X_attacked is not None
        and PERTURBABLE_IDX is not None
        and len(y) > 0
    ):
        X_clean_np = (
            X_clean.to_numpy(dtype=np.float32, copy=True)
            if isinstance(X_clean, pd.DataFrame)
            else np.asarray(X_clean, dtype=np.float32)
        )
        X_attacked_np = (
            X_attacked.to_numpy(dtype=np.float32, copy=True)
            if isinstance(X_attacked, pd.DataFrame)
            else np.asarray(X_attacked, dtype=np.float32)
        )

        delta = (
            X_attacked_np[:, PERTURBABLE_IDX]
            - X_clean_np[:, PERTURBABLE_IDX]
        )

        if delta.shape[1] > 0:
            mean_l2 = float(np.mean(np.linalg.norm(delta, axis=1)))
            mean_linf = float(np.mean(np.max(np.abs(delta), axis=1)))
            mean_changed = float(
                np.mean(np.sum(np.abs(delta) > 1e-8, axis=1))
            )

    return {
        "Clean Correct N": n_correct,
        "Clean Correct Positives": n_positive,
        "Successful Attacks": int(successful.sum()),
        "Successful Positive Attacks": int(successful_positive.sum()),
        "Attack Success Rate": (
            float(successful.sum() / n_correct)
            if n_correct else np.nan
        ),
        "Positive Attack Success": (
            float(successful_positive.sum() / n_positive)
            if n_positive else np.nan
        ),
        "Mean Probability Change": float(
            np.mean(attacked_probabilities - clean_probabilities)
        ) if len(y) else np.nan,
        "Mean |Probability Change|": float(
            np.mean(np.abs(attacked_probabilities - clean_probabilities))
        ) if len(y) else np.nan,
        "Mean L2": mean_l2,
        "Mean Linf": mean_linf,
        "Mean Changed Features": mean_changed,
    }


def build_defense_comparison_matrix(
    attack_examples,
    data_poisoning_examples,
    model_poisoning_examples,
    best_models,
    optimal_thresholds,
    adversarially_trained_models,
    ensemble_adversarial_models,
    adversarial_training_sets,
    ensemble_adversarial_training_sets,
    X_test,
    y_test,
    perturbable_cols,
    perturbable_idx,
    train_std,
    train_min,
    train_max,
    outlier_detector,
    evaluate_model_func,
    seed,
):
    """
    Build a unified defense comparison across three threat families:

      1. Evasion attacks that perturb test inputs.
      2. Data poisoning that corrupts training labels.
      3. Model poisoning that tampers with a fitted model artifact.

    Input-level defenses (feature squeezing, Isolation Forest rejection and
    randomized smoothing) are evaluated directly on every threat family.

    Model-level defenses are evaluated adaptively:
      * against evasion, a fresh TabularScore attack targets each robust model;
      * against data poisoning, the labels of the defense-specific augmented
        training set are poisoned and the estimator is retrained;
      * against model poisoning, the fitted robust artifact itself is tampered.

    For poisoning attacks, ASR/PAS quantify prediction degradation relative to
    the corresponding intact model under the same defense. Input perturbation
    norms are not applicable and are therefore NaN.
    """
    matrix_rows = []

    def append_row(
        *,
        model_name,
        threat_type,
        attack_name,
        defense_name,
        metrics,
        stats,
        metric_scope="full",
        rejection_rate=np.nan,
        coverage=np.nan,
        poison_fraction=np.nan,
        poison_strategy=None,
        severity=np.nan,
    ):
        matrix_rows.append(
            {
                "Model": model_name,
                "Threat Type": threat_type,
                "Attack": attack_name,
                "Defense": defense_name,
                "Metric Scope": metric_scope,
                "Poison Fraction": poison_fraction,
                "Poisoning Strategy": poison_strategy,
                "Severity": severity,
                "Rejection Rate": rejection_rate,
                "Coverage": coverage,
                **metrics,
                **stats,
            }
        )

    def evaluate_input_defenses(
        *,
        model_name,
        threat_type,
        attack_name,
        clean_model,
        attacked_model,
        X_clean,
        X_attacked,
        y,
        is_evasion,
        poison_fraction=np.nan,
        poison_strategy=None,
        severity=np.nan,
        local_seed=0,
    ):
        threshold = float(optimal_thresholds[model_name])
        y_arr = np.asarray(y).astype(int)

        # ------------------------------------------------------------
        # No defense
        # ------------------------------------------------------------
        clean_probs = clean_model.predict_proba(X_clean)[:, 1]
        attacked_probs = attacked_model.predict_proba(X_attacked)[:, 1]

        metrics = evaluate_probability_output(
            y_arr,
            attacked_probs,
            threshold,
        )
        stats = probability_shift_statistics(
            y_arr,
            clean_probs,
            attacked_probs,
            threshold,
            X_clean=X_clean if is_evasion else None,
            X_attacked=X_attacked if is_evasion else None,
            PERTURBABLE_IDX=perturbable_idx if is_evasion else None,
        )
        append_row(
            model_name=model_name,
            threat_type=threat_type,
            attack_name=attack_name,
            defense_name="None",
            metrics=metrics,
            stats=stats,
            poison_fraction=poison_fraction,
            poison_strategy=poison_strategy,
            severity=severity,
        )

        # ------------------------------------------------------------
        # Feature squeezing
        # ------------------------------------------------------------
        X_clean_sq = feature_squeeze(X_clean, perturbable_cols)
        X_attacked_sq = feature_squeeze(X_attacked, perturbable_cols)

        clean_sq_probs = clean_model.predict_proba(X_clean_sq)[:, 1]
        attacked_sq_probs = attacked_model.predict_proba(X_attacked_sq)[:, 1]

        metrics = evaluate_probability_output(
            y_arr,
            attacked_sq_probs,
            threshold,
        )
        stats = probability_shift_statistics(
            y_arr,
            clean_sq_probs,
            attacked_sq_probs,
            threshold,
            X_clean=X_clean_sq if is_evasion else None,
            X_attacked=X_attacked_sq if is_evasion else None,
            PERTURBABLE_IDX=perturbable_idx if is_evasion else None,
        )
        append_row(
            model_name=model_name,
            threat_type=threat_type,
            attack_name=attack_name,
            defense_name="Feature squeezing",
            metrics=metrics,
            stats=stats,
            poison_fraction=poison_fraction,
            poison_strategy=poison_strategy,
            severity=severity,
        )

        # ------------------------------------------------------------
        # Isolation Forest rejection
        # ------------------------------------------------------------
        flags = (
            outlier_detector
            .predict(X_attacked[perturbable_cols].astype(float))
            == -1
        )
        accepted = ~flags

        if accepted.any():
            X_clean_acc = X_clean.loc[accepted].copy()
            X_attacked_acc = X_attacked.loc[accepted].copy()
            y_acc = y_arr[accepted]

            clean_acc_probs = clean_model.predict_proba(X_clean_acc)[:, 1]
            attacked_acc_probs = attacked_model.predict_proba(
                X_attacked_acc
            )[:, 1]

            metrics = evaluate_probability_output(
                y_acc,
                attacked_acc_probs,
                threshold,
            )
            stats = probability_shift_statistics(
                y_acc,
                clean_acc_probs,
                attacked_acc_probs,
                threshold,
                X_clean=X_clean_acc if is_evasion else None,
                X_attacked=X_attacked_acc if is_evasion else None,
                PERTURBABLE_IDX=perturbable_idx if is_evasion else None,
            )
        else:
            metrics = evaluate_probability_output(
                np.array([], dtype=int),
                np.array([], dtype=float),
                threshold,
            )
            stats = probability_shift_statistics(
                np.array([], dtype=int),
                np.array([], dtype=float),
                np.array([], dtype=float),
                threshold,
            )

        append_row(
            model_name=model_name,
            threat_type=threat_type,
            attack_name=attack_name,
            defense_name="Isolation Forest rejection",
            metrics=metrics,
            stats=stats,
            metric_scope="accepted only",
            rejection_rate=float(flags.mean()),
            coverage=float(accepted.mean()),
            poison_fraction=poison_fraction,
            poison_strategy=poison_strategy,
            severity=severity,
        )

        # ------------------------------------------------------------
        # Randomized smoothing
        # ------------------------------------------------------------
        # The same RNG seed is deliberately used for intact and attacked
        # models so poisoning comparisons use the same noisy realizations.
        smooth_seed = seed + 10000 + local_seed
        clean_smooth_probs = randomized_smoothing_predict_proba(
            clean_model,
            X_clean,
            perturbable_idx,
            train_std,
            train_min,
            train_max,
            seed=smooth_seed,
        )
        attacked_smooth_probs = randomized_smoothing_predict_proba(
            attacked_model,
            X_attacked,
            perturbable_idx,
            train_std,
            train_min,
            train_max,
            seed=smooth_seed,
        )

        metrics = evaluate_probability_output(
            y_arr,
            attacked_smooth_probs,
            threshold,
        )
        stats = probability_shift_statistics(
            y_arr,
            clean_smooth_probs,
            attacked_smooth_probs,
            threshold,
            X_clean=X_clean if is_evasion else None,
            X_attacked=X_attacked if is_evasion else None,
            PERTURBABLE_IDX=perturbable_idx if is_evasion else None,
        )
        append_row(
            model_name=model_name,
            threat_type=threat_type,
            attack_name=attack_name,
            defense_name="Randomized smoothing",
            metrics=metrics,
            stats=stats,
            poison_fraction=poison_fraction,
            poison_strategy=poison_strategy,
            severity=severity,
        )

    # ================================================================
    # Input-level defenses against evasion attacks
    # ================================================================
    for scenario_idx, ((model_name, attack_name), sample) in enumerate(
        attack_examples.items()
    ):
        base_model = best_models[model_name]

        evaluate_input_defenses(
            model_name=model_name,
            threat_type="Evasion",
            attack_name=attack_name,
            clean_model=base_model,
            attacked_model=base_model,
            X_clean=sample["X_clean"],
            X_attacked=sample["X_adv"],
            y=sample["y"],
            is_evasion=True,
            local_seed=scenario_idx,
        )

    # ================================================================
    # Input-level defenses against data poisoning
    # ================================================================
    for scenario_idx, ((model_name, attack_name), sample) in enumerate(
        data_poisoning_examples.items()
    ):
        base_model = best_models[model_name]

        evaluate_input_defenses(
            model_name=model_name,
            threat_type="Data poisoning",
            attack_name=attack_name,
            clean_model=base_model,
            attacked_model=sample["attacked_model"],
            X_clean=sample["X_eval"],
            X_attacked=sample["X_eval"],
            y=sample["y_eval"],
            is_evasion=False,
            poison_fraction=float(sample["fraction"]),
            poison_strategy=sample["strategy_label"],
            local_seed=1000 + scenario_idx,
        )

    # ================================================================
    # Input-level defenses against model poisoning
    # ================================================================
    for scenario_idx, ((model_name, attack_name), sample) in enumerate(
        model_poisoning_examples.items()
    ):
        base_model = best_models[model_name]

        evaluate_input_defenses(
            model_name=model_name,
            threat_type="Model poisoning",
            attack_name=attack_name,
            clean_model=base_model,
            attacked_model=sample["attacked_model"],
            X_clean=sample["X_eval"],
            X_attacked=sample["X_eval"],
            y=sample["y_eval"],
            is_evasion=False,
            severity=float(sample["severity"]),
            local_seed=2000 + scenario_idx,
        )

    # ================================================================
    # Adaptive evasion evaluation of model-level defenses
    # ================================================================
    eval_n = min(500, len(X_test))
    eval_rng = np.random.default_rng(seed + 900)
    eval_idx = eval_rng.choice(
        len(X_test),
        size=eval_n,
        replace=False,
    )
    X_eval_subset = X_test.iloc[eval_idx].copy()
    y_eval_subset = np.asarray(y_test)[eval_idx].astype(int)

    def eval_adaptive_evasion(robust_models, defense_name, offset):
        for model_idx, (model_name, robust_model) in enumerate(
            robust_models.items()
        ):
            threshold = float(optimal_thresholds[model_name])
            X_clean, y_clean, _ = clean_correct_subset(
                robust_model,
                threshold,
                X_eval_subset,
                y_eval_subset,
            )

            if len(X_clean) == 0:
                continue

            X_adv = tabular_score_attack(
                robust_model,
                X_clean,
                y_clean,
                PERTURBABLE_IDX=perturbable_idx,
                TRAIN_STD=train_std,
                TRAIN_MIN=train_min,
                TRAIN_MAX=train_max,
                seed=seed + offset + model_idx,
            )

            clean_probs = robust_model.predict_proba(X_clean)[:, 1]
            adv_probs = robust_model.predict_proba(X_adv)[:, 1]

            metrics = evaluate_model_func(
                robust_model,
                threshold,
                X_adv,
                y_clean,
            )
            stats = probability_shift_statistics(
                y_clean,
                clean_probs,
                adv_probs,
                threshold,
                X_clean=X_clean,
                X_attacked=X_adv,
                PERTURBABLE_IDX=perturbable_idx,
            )

            append_row(
                model_name=model_name,
                threat_type="Evasion",
                attack_name="TabularScore / adaptive",
                defense_name=defense_name,
                metrics=metrics,
                stats=stats,
            )

    eval_adaptive_evasion(
        adversarially_trained_models,
        "Adversarial training",
        3000,
    )
    eval_adaptive_evasion(
        ensemble_adversarial_models,
        "Ensemble adversarial training",
        4000,
    )

    # ================================================================
    # Data poisoning against defense-specific training sets
    # ================================================================
    model_level_defenses = [
        (
            "Adversarial training",
            adversarially_trained_models,
            adversarial_training_sets,
        ),
        (
            "Ensemble adversarial training",
            ensemble_adversarial_models,
            ensemble_adversarial_training_sets,
        ),
    ]

    for scenario_idx, ((model_name, attack_name), sample) in enumerate(
        data_poisoning_examples.items()
    ):
        threshold = float(optimal_thresholds[model_name])
        X_eval = sample["X_eval"]
        y_eval = np.asarray(sample["y_eval"]).astype(int)

        for defense_idx, (
            defense_name,
            clean_robust_models,
            defense_training_sets,
        ) in enumerate(model_level_defenses):
            clean_robust_model = clean_robust_models[model_name]
            X_def_train, y_def_train = defense_training_sets[model_name]

            y_def_poisoned, _ = poison_training_labels(
                y_def_train,
                fraction=float(sample["fraction"]),
                strategy=sample["strategy_key"],
                seed=int(sample["seed"]) + 100 * (defense_idx + 1),
            )

            poisoned_robust_model = clone(best_models[model_name])
            poisoned_robust_model.fit(
                X_def_train,
                y_def_poisoned,
            )

            clean_probs = clean_robust_model.predict_proba(X_eval)[:, 1]
            poisoned_probs = poisoned_robust_model.predict_proba(
                X_eval
            )[:, 1]

            metrics = evaluate_model_func(
                poisoned_robust_model,
                threshold,
                X_eval,
                y_eval,
            )
            stats = probability_shift_statistics(
                y_eval,
                clean_probs,
                poisoned_probs,
                threshold,
            )

            append_row(
                model_name=model_name,
                threat_type="Data poisoning",
                attack_name=attack_name,
                defense_name=defense_name,
                metrics=metrics,
                stats=stats,
                poison_fraction=float(sample["fraction"]),
                poison_strategy=sample["strategy_label"],
            )

    # ================================================================
    # Model poisoning of robust model artifacts
    # ================================================================
    for scenario_idx, ((model_name, attack_name), sample) in enumerate(
        model_poisoning_examples.items()
    ):
        threshold = float(optimal_thresholds[model_name])
        X_eval = sample["X_eval"]
        y_eval = np.asarray(sample["y_eval"]).astype(int)

        for defense_name, clean_robust_models, _ in model_level_defenses:
            clean_robust_model = clean_robust_models[model_name]
            poisoned_robust_model = poison_model_artifact(
                model=clean_robust_model,
                model_name=model_name,
                severity=float(sample["severity"]),
            )

            clean_probs = clean_robust_model.predict_proba(X_eval)[:, 1]
            poisoned_probs = poisoned_robust_model.predict_proba(
                X_eval
            )[:, 1]

            metrics = evaluate_model_func(
                poisoned_robust_model,
                threshold,
                X_eval,
                y_eval,
            )
            stats = probability_shift_statistics(
                y_eval,
                clean_probs,
                poisoned_probs,
                threshold,
            )

            append_row(
                model_name=model_name,
                threat_type="Model poisoning",
                attack_name=attack_name,
                defense_name=defense_name,
                metrics=metrics,
                stats=stats,
                severity=float(sample["severity"]),
            )

    return pd.DataFrame(matrix_rows)