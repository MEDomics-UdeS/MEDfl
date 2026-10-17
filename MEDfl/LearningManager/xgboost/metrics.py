import math
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def _safe_float(value: Any) -> float:
    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return float("nan")

        return value
    except Exception:
        return float("nan")


def binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": _safe_float(
            accuracy_score(y_true, y_pred)
        ),
    }

    try:
        metrics["auc"] = _safe_float(
            roc_auc_score(y_true, y_prob)
        )
    except Exception:
        metrics["auc"] = float("nan")

    try:
        metrics["logloss"] = _safe_float(
            log_loss(y_true, y_prob)
        )
    except Exception:
        metrics["logloss"] = float("nan")

    return metrics


def regression_metrics(
    y_true,
    y_pred,
    num_features: int = 0,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = mean_squared_error(
        y_true,
        y_pred,
        squared=False,
    )
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    p = int(num_features)

    if n > p + 1:
        adjusted_r2 = (
            1.0
            - (1.0 - r2)
            * ((n - 1.0) / (n - p - 1.0))
        )
    else:
        adjusted_r2 = float("nan")

    return {
        "rmse": _safe_float(rmse),
        "mae": _safe_float(mae),
        "r2": _safe_float(r2),
        "adjusted_r2": _safe_float(adjusted_r2),
    }


def multiclass_metrics(
    y_true,
    y_prob,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "accuracy": _safe_float(
            accuracy_score(y_true, y_pred)
        ),
    }

    try:
        metrics["macro_f1"] = _safe_float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
            )
        )
    except Exception:
        metrics["macro_f1"] = float("nan")

    try:
        metrics["mlogloss"] = _safe_float(
            log_loss(y_true, y_prob)
        )
    except Exception:
        metrics["mlogloss"] = float("nan")

    return metrics


def evaluate_predictions(
    task: str,
    y_true,
    y_pred_or_prob,
    threshold: float = 0.5,
    num_features: int = 0,
) -> Dict[str, float]:
    if task == "binary":
        return binary_metrics(
            y_true,
            y_pred_or_prob,
            threshold=threshold,
        )

    if task == "regression":
        return regression_metrics(
            y_true,
            y_pred_or_prob,
            num_features=num_features,
        )

    if task == "multiclass":
        return multiclass_metrics(
            y_true,
            y_pred_or_prob,
        )

    raise ValueError(f"Unsupported task: {task}")


def aggregate_weighted_metrics(results):
    aggregated = {}

    keys = set()
    for _, metrics in results:
        keys.update(metrics.keys())

    for key in keys:
        weighted_sum = 0.0
        contributing_examples = 0

        for num_examples, metrics in results:
            if key not in metrics:
                continue

            try:
                value = float(metrics[key])
            except Exception:
                continue

            if math.isnan(value) or math.isinf(value):
                continue

            weighted_sum += num_examples * value
            contributing_examples += num_examples

        if contributing_examples > 0:
            aggregated[key] = (
                weighted_sum / contributing_examples
            )

    return aggregated