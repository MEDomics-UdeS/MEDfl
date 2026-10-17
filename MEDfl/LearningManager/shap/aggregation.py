from __future__ import annotations

from typing import Any

import numpy as np


def _validate_client_results(
    client_results: list[dict],
) -> tuple[list[str], int]:
    if not client_results:
        raise ValueError(
            "No client SHAP results were provided"
        )

    reference_features = client_results[0][
        "feature_names"
    ]

    feature_count = len(reference_features)

    required_array_keys = {
        "abs_sum",
        "signed_sum",
        "square_sum",
        "positive_count",
        "negative_count",
        "zero_count",
    }

    for result in client_results:
        if result["feature_names"] != reference_features:
            raise ValueError(
                "Clients used different feature names or orders"
            )

        if int(result["sample_count"]) <= 0:
            raise ValueError(
                f"Client {result.get('client_id')} returned an "
                "invalid SHAP sample count"
            )

        for key in required_array_keys:
            if key not in result:
                raise ValueError(
                    f"Client result is missing '{key}'"
                )

            values = np.asarray(result[key])

            if values.shape != (feature_count,):
                raise ValueError(
                    f"Client {result.get('client_id')} returned "
                    f"invalid shape {values.shape} for {key}. "
                    f"Expected ({feature_count},)."
                )

    return reference_features, feature_count


def aggregate_federated_shap(
    client_results: list[dict],
    include_client_results: bool = True,
) -> dict[str, Any]:
    """
    Aggregate additive local SHAP statistics.

    The global mean absolute SHAP is equivalent to calculating:

        sum of all |SHAP values| / number of all explained samples
    """

    feature_names, feature_count = (
        _validate_client_results(client_results)
    )

    total_samples = sum(
        int(result["sample_count"])
        for result in client_results
    )

    abs_sum = np.zeros(
        feature_count,
        dtype=np.float64,
    )
    signed_sum = np.zeros(
        feature_count,
        dtype=np.float64,
    )
    square_sum = np.zeros(
        feature_count,
        dtype=np.float64,
    )
    positive_count = np.zeros(
        feature_count,
        dtype=np.float64,
    )
    negative_count = np.zeros(
        feature_count,
        dtype=np.float64,
    )
    zero_count = np.zeros(
        feature_count,
        dtype=np.float64,
    )

    for result in client_results:
        abs_sum += np.asarray(
            result["abs_sum"],
            dtype=np.float64,
        )
        signed_sum += np.asarray(
            result["signed_sum"],
            dtype=np.float64,
        )
        square_sum += np.asarray(
            result["square_sum"],
            dtype=np.float64,
        )
        positive_count += np.asarray(
            result["positive_count"],
            dtype=np.float64,
        )
        negative_count += np.asarray(
            result["negative_count"],
            dtype=np.float64,
        )
        zero_count += np.asarray(
            result["zero_count"],
            dtype=np.float64,
        )

    mean_abs = abs_sum / total_samples
    mean_signed = signed_sum / total_samples
    mean_square = square_sum / total_samples

    # Var(X) = E(X²) - E(X)²
    variance = np.maximum(
        mean_square - np.square(mean_signed),
        0.0,
    )

    standard_deviation = np.sqrt(variance)

    features = []

    for index, feature_name in enumerate(
        feature_names
    ):
        features.append(
            {
                "feature": feature_name,
                "feature_index": index,
                "mean_abs_shap": float(
                    mean_abs[index]
                ),
                "mean_signed_shap": float(
                    mean_signed[index]
                ),
                "shap_std": float(
                    standard_deviation[index]
                ),
                "positive_rate": float(
                    positive_count[index]
                    / total_samples
                ),
                "negative_rate": float(
                    negative_count[index]
                    / total_samples
                ),
                "zero_rate": float(
                    zero_count[index]
                    / total_samples
                ),
            }
        )

    # Highest global importance first.
    features.sort(
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )

    final_result: dict[str, Any] = {
        "status": "completed",
        "participating_clients": len(
            client_results
        ),
        "explained_samples": int(
            total_samples
        ),
        "feature_importance": features,
    }

    if include_client_results:
        client_summaries = []

        for result in client_results:
            sample_count = int(
                result["sample_count"]
            )

            client_mean_abs = (
                np.asarray(
                    result["abs_sum"],
                    dtype=np.float64,
                )
                / sample_count
            )

            local_features = [
                {
                    "feature": feature_name,
                    "feature_index": index,
                    "mean_abs_shap": float(
                        client_mean_abs[index]
                    ),
                }
                for index, feature_name in enumerate(
                    feature_names
                )
            ]

            local_features.sort(
                key=lambda item: item[
                    "mean_abs_shap"
                ],
                reverse=True,
            )

            client_summaries.append(
                {
                    "client_id": result[
                        "client_id"
                    ],
                    "explained_samples": (
                        sample_count
                    ),
                    "feature_importance": (
                        local_features
                    ),
                }
            )

        final_result["client_results"] = (
            client_summaries
        )

    return final_result