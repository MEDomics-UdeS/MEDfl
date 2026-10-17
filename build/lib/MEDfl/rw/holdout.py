"""
Federated holdout test for MEDfl real-world pipelines.

After the last round, every client tests the final global model on its own
local holdout CSV (a file never used for training). Clients only send
aggregate numbers back to the server:

    - the confusion matrix at the decision threshold,
    - their local metrics,
    - two histograms of the predicted probabilities (positives / negatives).

The server sums the confusion matrices to get exact pooled metrics
(sensitivity, specificity, PPV, NPV, F1, accuracy) and sums the histograms
to rebuild a pooled ROC curve and AUC. The pooled AUC is exact up to the
histogram bin width: two scores falling in the same bin count as a tie.

The holdout goes through the same preprocessing as the client training
data: the server features in the server order, the raw values (the
real-world clients do not impute or scale), and a binary 0/1 target.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Statuses a client can report in metrics["holdout_status"]
HOLDOUT_COMPLETED = "completed"
HOLDOUT_SKIPPED = "skipped"  # fewer rows than minimum_samples
HOLDOUT_MISSING = "missing"  # no holdout file on this client
HOLDOUT_FAILED = "failed"
HOLDOUT_UNSUPPORTED = "unsupported_task"


@dataclass
class HoldoutConfig:
    """
    Configuration for the optional post-training federated holdout test.
    """

    enabled: bool = False

    # A prediction is positive when its probability is >= threshold
    threshold: float = 0.5

    # Number of probability bins used to rebuild the pooled ROC curve / AUC
    num_bins: int = 100

    # Clients with fewer holdout rows do not send results
    minimum_samples: int = 10

    include_client_results: bool = True

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Holdout enabled must be a boolean")

        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("Holdout threshold must be between 0 and 1")

        if int(self.num_bins) < 2:
            raise ValueError("Holdout num_bins must be at least 2")

        if int(self.minimum_samples) <= 0:
            raise ValueError("Holdout minimum_samples must be greater than zero")

    def to_evaluate_config(self) -> Dict[str, Any]:
        """
        Keys added to the evaluate config of the final round.
        Flower configs are Scalar-only.
        """
        return {
            "holdout_enabled": True,
            "holdout_threshold": float(self.threshold),
            "holdout_num_bins": int(self.num_bins),
            "holdout_minimum_samples": int(self.minimum_samples),
        }


# ----------------------------------------------------------------------
# Client side
# ----------------------------------------------------------------------


def load_holdout(
    path: str,
    features: List[str],
    target: str,
    drop_missing_features: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Read a holdout CSV with the training schema.

    Rows without a target are dropped. Rows with a missing feature value are
    dropped when drop_missing_features is True (the neural network cannot
    predict them); XGBoost keeps them since it handles missing values.

    Returns (X, y, dataset information).
    """
    data = pd.read_csv(path)

    if target not in data.columns:
        raise ValueError(
            f"The holdout dataset has no '{target}' column (the model target)"
        )

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(
            f"The holdout dataset is missing {len(missing)} training "
            f"column(s): {', '.join(missing)}"
        )

    X = data[features].apply(pd.to_numeric, errors="coerce")
    non_numeric = [
        column
        for column in features
        if data[column].notna().any() and X[column].isna().all()
    ]
    if non_numeric:
        raise ValueError(
            f"These holdout columns are not numeric: {', '.join(non_numeric)}"
        )

    y = pd.to_numeric(data[target], errors="coerce")
    has_target = y.notna().values
    has_features = X.notna().all(axis=1).values

    keep = has_target & has_features if drop_missing_features else has_target
    X, y = X.values[keep], y.values[keep].astype(float)

    classes = set(np.unique(y).tolist())
    if not classes.issubset({0.0, 1.0}):
        raise ValueError(
            f"The target '{target}' must be binary (0/1) in the holdout "
            f"dataset, found {len(classes)} values: {sorted(classes)[:10]}"
        )

    info = {
        "name": os.path.basename(path),
        "rows_in_file": int(len(data)),
        "rows_without_target": int((~has_target).sum()),
        "rows_with_missing_features": int(
            (has_target & ~has_features).sum()
        ),
        "rows_missing_features_dropped": bool(drop_missing_features),
    }
    return X.astype(np.float32), y, info


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    """None when undefined (e.g. no positive case), so JSON stays valid."""
    return float(numerator) / float(denominator) if denominator else None


def _confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, int]:
    y_pred = (y_prob >= threshold).astype(float)
    return {
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
    }


def confusion_metrics(confusion: Dict[str, int]) -> Dict[str, Optional[float]]:
    """Threshold metrics from a confusion matrix (same definitions as the simulation holdout)."""
    tp, fp, fn, tn = (confusion[key] for key in ("TP", "FP", "FN", "TN"))
    return {
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        # same value as 2 * sensitivity * ppv / (sensitivity + ppv), and 0 when there is no true positive
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _bin_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(num_bins) + 1)


def local_holdout_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    num_bins: int,
    minimum_samples: int,
    client_id: str,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summary of one client's holdout test, safe to send to the server.
    Below minimum_samples rows, only the row count is sent.
    """
    rows = int(len(y_true))
    positives = int(y_true.sum())
    summary = {
        "client_id": client_id,
        "dataset": {**dataset, "rows": rows, "positives": positives, "negatives": rows - positives},
    }

    if rows < int(minimum_samples):
        return {
            **summary,
            "status": HOLDOUT_SKIPPED,
            "reason": (
                f"{rows} usable holdout row(s), fewer than the minimum of "
                f"{int(minimum_samples)}"
            ),
        }

    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    confusion = _confusion(y_true, y_prob, threshold)
    edges = _bin_edges(num_bins)

    metrics = {
        "auc": (
            float(roc_auc_score(y_true, y_prob))
            if 0 < positives < rows
            else None
        ),
        **confusion_metrics(confusion),
    }

    return {
        **summary,
        "status": HOLDOUT_COMPLETED,
        "threshold": float(threshold),
        "num_bins": int(num_bins),
        "metrics": metrics,
        "confusion_matrix": confusion,
        "pos_hist": np.histogram(y_prob[y_true == 1], bins=edges)[0].astype(int).tolist(),
        "neg_hist": np.histogram(y_prob[y_true == 0], bins=edges)[0].astype(int).tolist(),
    }


def run_local_holdout(
    config: Dict[str, Any],
    holdout_path: Optional[str],
    features: List[str],
    target: str,
    predict: Callable[[np.ndarray], np.ndarray],
    client_id: str,
    drop_missing_features: bool,
) -> Dict[str, Any]:
    """
    Run the holdout test on a client and return the entries to add to the
    Flower evaluate metrics: holdout_status, then holdout_result (JSON) or
    holdout_error.

    predict maps the holdout features to positive-class probabilities with
    the final global model.
    """
    if not holdout_path or not os.path.exists(holdout_path):
        print(f"[Client/Holdout] No holdout file on this client ({holdout_path})", flush=True)
        return {
            "holdout_status": HOLDOUT_MISSING,
            "holdout_client_id": client_id,
            "holdout_error": "No holdout file on this client",
        }

    try:
        X, y, dataset = load_holdout(holdout_path, features, target, drop_missing_features)
        probabilities = predict(X) if len(y) else np.empty((0,))
        summary = local_holdout_summary(
            y_true=y,
            y_prob=probabilities,
            threshold=float(config.get("holdout_threshold", 0.5)),
            num_bins=int(config.get("holdout_num_bins", 100)),
            minimum_samples=int(config.get("holdout_minimum_samples", 10)),
            client_id=client_id,
            dataset=dataset,
        )
        print(
            f"[Client/Holdout] {summary['status']} | client={client_id} | "
            f"rows={summary['dataset']['rows']} | metrics={summary.get('metrics')}",
            flush=True,
        )
        return {
            "holdout_status": summary["status"],
            "holdout_client_id": client_id,
            "holdout_result": json.dumps(summary, allow_nan=False),
        }
    except Exception as exc:
        print(f"[Client/Holdout] Holdout test failed: {exc}", flush=True)
        return {
            "holdout_status": HOLDOUT_FAILED,
            "holdout_client_id": client_id,
            "holdout_error": str(exc),
        }


# ----------------------------------------------------------------------
# Server side
# ----------------------------------------------------------------------


def pooled_roc_auc(pos_hist: np.ndarray, neg_hist: np.ndarray) -> Tuple[Optional[float], Optional[Dict[str, List[float]]]]:
    """
    ROC curve and AUC from score histograms.

    The thresholds are the bin edges. The AUC is the Mann-Whitney statistic
    where two scores in the same bin count as a tie (half a pair).
    """
    positives, negatives = pos_hist.sum(), neg_hist.sum()
    if positives == 0 or negatives == 0:
        return None, None

    # positives in strictly higher bins than each bin
    positives_above = np.concatenate([np.cumsum(pos_hist[::-1])[::-1][1:], [0]])
    auc = float((neg_hist * (positives_above + 0.5 * pos_hist)).sum() / (positives * negatives))

    # highest threshold first, starting at (0, 0)
    tpr = np.concatenate([[0.0], np.cumsum(pos_hist[::-1]) / positives])
    fpr = np.concatenate([[0.0], np.cumsum(neg_hist[::-1]) / negatives])
    return auc, {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}


def _client_name(client_proxy) -> str:
    return str(getattr(client_proxy, "cid", client_proxy))


def aggregate_federated_holdout(
    results,
    failures,
    holdout_config: HoldoutConfig,
    backend: str,
) -> Dict[str, Any]:
    """
    Aggregate the clients' holdout summaries of the final round.

    Args:
        results: Flower evaluate results, a list of (client_proxy, EvaluateRes)
        failures: Flower evaluate failures
        holdout_config: the server HoldoutConfig
        backend: "pytorch" or "xgboost"

    The "pooled" entry has the same keys as the simulation holdout results
    (dataset, threshold, metrics, confusion_matrix, roc_curve).
    """
    completed, skipped, missing, failed = [], [], [], []

    for client_proxy, eval_res in results:
        metrics = eval_res.metrics or {}
        status = metrics.get("holdout_status")
        # the client hostname when the client sent it, else the Flower client id
        name = str(metrics.get("holdout_client_id") or _client_name(client_proxy))

        summary = None
        if metrics.get("holdout_result"):
            try:
                summary = json.loads(metrics["holdout_result"])
            except Exception as exc:
                failed.append({"client_id": name, "error": f"Invalid holdout JSON: {exc}"})
                continue

        if status == HOLDOUT_COMPLETED and summary:
            if int(summary.get("num_bins", -1)) != int(holdout_config.num_bins):
                failed.append({
                    "client_id": summary.get("client_id", name),
                    "error": f"Used {summary.get('num_bins')} histogram bins instead of {holdout_config.num_bins}",
                })
                continue
            completed.append(summary)
        elif status == HOLDOUT_SKIPPED and summary:
            skipped.append({
                "client_id": summary.get("client_id", name),
                "rows": summary["dataset"]["rows"],
                "reason": summary.get("reason"),
            })
        elif status == HOLDOUT_MISSING:
            missing.append({"client_id": name, "error": metrics.get("holdout_error")})
        else:
            failed.append({
                "client_id": name,
                "error": metrics.get("holdout_error", "No holdout result returned"),
            })

    for failure in failures or []:
        if isinstance(failure, BaseException):
            failed.append({"client_id": None, "error": str(failure)})
        else:
            failed.append({"client_id": _client_name(failure[0]), "error": "Evaluation failed"})

    result: Dict[str, Any] = {
        "backend": backend,
        "threshold": float(holdout_config.threshold),
        "num_bins": int(holdout_config.num_bins),
        "minimum_samples": int(holdout_config.minimum_samples),
        "participating_clients": len(completed),
        "skipped_clients": skipped,
        "missing_clients": missing,
        "failed_clients": failed,
    }

    if not completed:
        return {
            **result,
            "status": "failed",
            "error": "No client returned a holdout result",
        }

    confusion = {
        key: int(sum(summary["confusion_matrix"][key] for summary in completed))
        for key in ("TP", "FP", "FN", "TN")
    }
    pos_hist = np.sum([summary["pos_hist"] for summary in completed], axis=0)
    neg_hist = np.sum([summary["neg_hist"] for summary in completed], axis=0)
    auc, roc = pooled_roc_auc(pos_hist, neg_hist)

    rows = sum(summary["dataset"]["rows"] for summary in completed)
    positives = sum(summary["dataset"]["positives"] for summary in completed)

    weighted = [
        (summary["dataset"]["rows"], summary["metrics"]["auc"])
        for summary in completed
        if summary["metrics"].get("auc") is not None
    ]
    auc_weighted_mean = (
        sum(n * value for n, value in weighted) / sum(n for n, _ in weighted)
        if weighted
        else None
    )

    result.update({
        "status": "completed",
        "pooled": {
            "dataset": {
                "name": f"Federated holdout ({len(completed)} client(s))",
                "rows": int(rows),
                "positives": int(positives),
                "negatives": int(rows - positives),
                "rows_without_target": int(
                    sum(summary["dataset"]["rows_without_target"] for summary in completed)
                ),
            },
            "threshold": float(holdout_config.threshold),
            "metrics": {"auc": auc, **confusion_metrics(confusion)},
            "confusion_matrix": confusion,
            "roc_curve": roc,
        },
        "auc_method": f"pooled from {int(holdout_config.num_bins)}-bin score histograms",
        "auc_weighted_mean": auc_weighted_mean,
    })

    if holdout_config.include_client_results:
        result["clients"] = [
            {
                "client_id": summary["client_id"],
                "dataset": summary["dataset"],
                "metrics": summary["metrics"],
                "confusion_matrix": summary["confusion_matrix"],
            }
            for summary in completed
        ]

    return result


# The app reads the result from this log line (the XGBoost server answers
# its caller before training ends)
HOLDOUT_LOG_PREFIX = "[Server/Holdout] FEDERATED_HOLDOUT_RESULT "


def finalize_federated_holdout(
    results,
    failures,
    holdout_config: HoldoutConfig,
    backend: str,
) -> Dict[str, Any]:
    """
    Aggregate the final-round holdout results without ever stopping the
    server, then print them on one line prefixed by HOLDOUT_LOG_PREFIX.
    """
    try:
        result = aggregate_federated_holdout(results, failures, holdout_config, backend)
    except Exception as exc:
        result = {"status": "failed", "backend": backend, "error": f"Holdout aggregation failed: {exc}"}

    print(
        f"[Server/Holdout] Federated holdout {result['status']} | "
        f"clients={result.get('participating_clients', 0)} | "
        f"metrics={(result.get('pooled') or {}).get('metrics')}",
        flush=True,
    )
    print(HOLDOUT_LOG_PREFIX + json.dumps(result, allow_nan=False, default=str), flush=True)
    return result
