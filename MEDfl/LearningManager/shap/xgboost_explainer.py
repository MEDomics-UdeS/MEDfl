from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader


logger = logging.getLogger("MEDfl.shap.xgboost")


class LocalXGBoostSHAPExplainer:
    """
    Calculate local SHAP values for a federated XGBoost client.

    The final federated XGBoost booster is explained separately on
    every client's selected local partition.

    Raw samples and raw per-sample SHAP values are never returned.
    Only additive summary statistics are returned so the existing
    federated SHAP aggregation function can be reused.
    """

    def __init__(
        self,
        background_size: int = 100,
        explanation_size: int = 500,
        random_seed: int = 42,
        clipping_value: Optional[float] = None,
        model_output: str = "raw",
    ) -> None:
        self.background_size = int(background_size)
        self.explanation_size = int(explanation_size)
        self.random_seed = int(random_seed)
        self.clipping_value = clipping_value
        self.model_output = str(model_output)

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if self.background_size <= 0:
            raise ValueError(
                "background_size must be greater than zero"
            )

        if self.explanation_size <= 0:
            raise ValueError(
                "explanation_size must be greater than zero"
            )

        if self.clipping_value is not None:
            if self.clipping_value <= 0:
                raise ValueError(
                    "clipping_value must be greater than zero"
                )

        if self.model_output not in {
            "raw",
            "probability",
        }:
            raise ValueError(
                "model_output must be 'raw' or 'probability'"
            )

    @staticmethod
    def _collect_features(
        loader: DataLoader,
    ) -> np.ndarray:
        """
        Collect only the input features from a client DataLoader.

        Labels are not needed for SHAP calculation.
        """

        if loader is None:
            raise ValueError(
                "Cannot calculate XGBoost SHAP using a None loader"
            )

        feature_batches: List[np.ndarray] = []

        for batch_index, batch in enumerate(loader):
            if not isinstance(batch, (tuple, list)):
                raise TypeError(
                    "Each DataLoader batch must be a tuple or list"
                )

            if len(batch) == 0:
                raise ValueError(
                    f"Empty batch received at index {batch_index}"
                )

            features = batch[0]

            if isinstance(features, torch.Tensor):
                features = (
                    features
                    .detach()
                    .cpu()
                    .numpy()
                )

            features = np.asarray(features)

            if features.ndim == 1:
                features = features.reshape(1, -1)

            if features.ndim != 2:
                raise ValueError(
                    "The XGBoost SHAP implementation supports "
                    "two-dimensional tabular inputs. "
                    f"Received shape {features.shape}."
                )

            if not np.issubdtype(
                features.dtype,
                np.number,
            ):
                try:
                    features = features.astype(
                        np.float32
                    )
                except Exception as exc:
                    raise TypeError(
                        "XGBoost SHAP features must be numeric"
                    ) from exc

            features = features.astype(
                np.float32,
                copy=False,
            )

            if np.isnan(features).any():
                raise ValueError(
                    "XGBoost SHAP input contains NaN values "
                    f"in batch {batch_index}"
                )

            if np.isinf(features).any():
                raise ValueError(
                    "XGBoost SHAP input contains infinite values "
                    f"in batch {batch_index}"
                )

            feature_batches.append(features)

        if not feature_batches:
            raise ValueError(
                "Cannot calculate XGBoost SHAP using an empty "
                "DataLoader"
            )

        collected_features = np.concatenate(
            feature_batches,
            axis=0,
        )

        if collected_features.ndim != 2:
            raise ValueError(
                "Unexpected collected XGBoost feature shape: "
                f"{collected_features.shape}"
            )

        return collected_features

    @staticmethod
    def _sample_rows(
        data: np.ndarray,
        requested_size: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        selected_size = min(
            int(requested_size),
            int(data.shape[0]),
        )

        indices = random_generator.choice(
            data.shape[0],
            size=selected_size,
            replace=False,
        )

        return data[indices]

    @staticmethod
    def _normalize_shap_values(
        raw_values: Any,
        sample_count: int,
        feature_count: int,
    ) -> np.ndarray:
        """
        Normalize SHAP output to:

            (number_of_samples, number_of_features)

        This first implementation supports binary classification and
        regression. Multiclass SHAP requires class-specific aggregation.
        """

        if hasattr(raw_values, "values"):
            raw_values = raw_values.values

        if isinstance(raw_values, list):
            if len(raw_values) != 1:
                raise ValueError(
                    "Multiclass XGBoost SHAP is not yet supported. "
                    "The returned SHAP result contains several "
                    "class-specific arrays."
                )

            raw_values = raw_values[0]

        shap_values = np.asarray(raw_values)

        # Possible binary/single-output shape:
        # (samples, features, 1)
        if (
            shap_values.ndim == 3
            and shap_values.shape[-1] == 1
        ):
            shap_values = shap_values[..., 0]

        # Possible shape:
        # (1, samples, features)
        if (
            shap_values.ndim == 3
            and shap_values.shape[0] == 1
        ):
            shap_values = shap_values[0]

        if shap_values.ndim == 3:
            raise ValueError(
                "Multiclass XGBoost SHAP is not yet supported. "
                "Class-specific SHAP values were returned with "
                f"shape {shap_values.shape}."
            )

        expected_shape = (
            sample_count,
            feature_count,
        )

        if shap_values.shape != expected_shape:
            raise ValueError(
                "Unexpected XGBoost SHAP output shape. "
                f"Expected {expected_shape}, "
                f"received {shap_values.shape}."
            )

        shap_values = shap_values.astype(
            np.float64,
            copy=False,
        )

        if np.isnan(shap_values).any():
            raise ValueError(
                "Calculated XGBoost SHAP values contain NaN values"
            )

        if np.isinf(shap_values).any():
            raise ValueError(
                "Calculated XGBoost SHAP values contain "
                "infinite values"
            )

        return shap_values

    def _create_explainer(
        self,
        booster: xgb.Booster,
        background: np.ndarray,
    ):
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is enabled for XGBoost, but the shap "
                "package is not installed."
            ) from exc

        if not isinstance(booster, xgb.Booster):
            raise TypeError(
                "LocalXGBoostSHAPExplainer expects an "
                "xgboost.Booster"
            )

        try:
            return shap.TreeExplainer(
                model=booster,
                data=background,
                feature_perturbation="interventional",
                model_output=self.model_output,
            )

        except Exception as exc:
            # Some SHAP/XGBoost combinations may not support every
            # TreeExplainer argument. Keep a conservative fallback.
            logger.warning(
                "Could not initialize TreeExplainer with the "
                "configured background and model output. "
                "Falling back to the default TreeExplainer: %s",
                exc,
            )

            return shap.TreeExplainer(
                model=booster,
            )

    @staticmethod
    def _calculate_raw_values(
        explainer,
        explanation_data: np.ndarray,
    ):
        """
        Support different SHAP versions.
        """

        try:
            return explainer.shap_values(
                explanation_data,
                check_additivity=False,
            )

        except TypeError:
            # Older SHAP versions may not accept
            # check_additivity here.
            return explainer.shap_values(
                explanation_data
            )

    def calculate(
        self,
        booster: xgb.Booster,
        loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Calculate local XGBoost SHAP additive summaries.
        """

        features = self._collect_features(loader)

        if len(features) < 2:
            raise ValueError(
                "At least two samples are required for "
                "XGBoost SHAP"
            )

        random_generator = np.random.default_rng(
            self.random_seed
        )

        background = self._sample_rows(
            data=features,
            requested_size=self.background_size,
            random_generator=random_generator,
        )

        explanation_data = self._sample_rows(
            data=features,
            requested_size=self.explanation_size,
            random_generator=random_generator,
        )

        logger.info(
            "Calculating local XGBoost SHAP | "
            "available=%d | background=%d | explained=%d | "
            "model_output=%s",
            len(features),
            len(background),
            len(explanation_data),
            self.model_output,
        )

        explainer = self._create_explainer(
            booster=booster,
            background=background,
        )

        raw_values = self._calculate_raw_values(
            explainer=explainer,
            explanation_data=explanation_data,
        )

        shap_values = self._normalize_shap_values(
            raw_values=raw_values,
            sample_count=len(explanation_data),
            feature_count=explanation_data.shape[1],
        )

        if self.clipping_value is not None:
            shap_values = np.clip(
                shap_values,
                -self.clipping_value,
                self.clipping_value,
            )

        result: Dict[str, Any] = {
            "sample_count": int(
                shap_values.shape[0]
            ),

            "abs_sum": np.sum(
                np.abs(shap_values),
                axis=0,
            ),

            "signed_sum": np.sum(
                shap_values,
                axis=0,
            ),

            "square_sum": np.sum(
                np.square(shap_values),
                axis=0,
            ),

            "positive_count": np.sum(
                shap_values > 0,
                axis=0,
            ).astype(np.int64),

            "negative_count": np.sum(
                shap_values < 0,
                axis=0,
            ).astype(np.int64),

            "zero_count": np.sum(
                shap_values == 0,
                axis=0,
            ).astype(np.int64),
        }

        return result