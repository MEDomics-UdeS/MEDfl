from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


logger = logging.getLogger("MEDfl.shap")


class LocalPyTorchSHAPExplainer:
    """
    Calculate SHAP values for one client partition.

    Raw samples and per-sample SHAP values are not returned.
    Only additive summary statistics are returned.
    """

    def __init__(
        self,
        explainer_type: str = "gradient",
        background_size: int = 100,
        explanation_size: int = 500,
        random_seed: int = 42,
        clipping_value: float | None = None,
    ) -> None:
        self.explainer_type = explainer_type
        self.background_size = background_size
        self.explanation_size = explanation_size
        self.random_seed = random_seed
        self.clipping_value = clipping_value

    @staticmethod
    def _collect_features(
        loader: DataLoader,
    ) -> torch.Tensor:
        """
        Collect only X values from the selected client loader.
        Labels are not required to calculate SHAP values.
        """
        feature_batches: list[torch.Tensor] = []

        for batch_index, batch in enumerate(loader):
            if not isinstance(batch, (tuple, list)):
                raise TypeError(
                    "Each DataLoader batch must be a tuple or list"
                )

            if len(batch) == 0:
                raise ValueError(
                    f"Empty batch received at index {batch_index}"
                )

            X = batch[0]

            if not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X)

            X = X.detach().cpu().float()

            if torch.isnan(X).any():
                raise ValueError(
                    f"SHAP input contains NaN values in batch "
                    f"{batch_index}"
                )

            if torch.isinf(X).any():
                raise ValueError(
                    f"SHAP input contains infinite values in batch "
                    f"{batch_index}"
                )

            feature_batches.append(X)

        if not feature_batches:
            raise ValueError(
                "Cannot calculate SHAP using an empty DataLoader"
            )

        features = torch.cat(
            feature_batches,
            dim=0,
        )

        # First implementation: tabular data.
        if features.ndim != 2:
            raise ValueError(
                "The current SHAP implementation supports tabular "
                "input with shape (samples, features). "
                f"Received shape {tuple(features.shape)}."
            )

        return features

    @staticmethod
    def _sample_rows(
        data: torch.Tensor,
        requested_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        selected_size = min(
            requested_size,
            len(data),
        )

        indices = torch.randperm(
            len(data),
            generator=generator,
        )[:selected_size]

        return data[indices]

    @staticmethod
    def _normalize_shap_values(
        raw_values: Any,
        sample_count: int,
        feature_count: int,
    ) -> np.ndarray:
        """
        Normalize the different shapes returned by SHAP versions.

        Expected final shape:
            (number_of_samples, number_of_features)
        """

        if hasattr(raw_values, "values"):
            raw_values = raw_values.values

        if isinstance(raw_values, list):
            if len(raw_values) != 1:
                raise ValueError(
                    "The current implementation supports binary or "
                    "single-output models only. Multiclass SHAP "
                    "requires class-specific aggregation."
                )

            raw_values = raw_values[0]

        values = np.asarray(raw_values)

        # Possible shape: (samples, features, 1)
        if (
            values.ndim == 3
            and values.shape[-1] == 1
        ):
            values = values[..., 0]

        # Possible shape: (1, samples, features)
        if (
            values.ndim == 3
            and values.shape[0] == 1
        ):
            values = values[0]

        expected_shape = (
            sample_count,
            feature_count,
        )

        if values.shape != expected_shape:
            raise ValueError(
                "Unexpected SHAP output shape. "
                f"Expected {expected_shape}, received {values.shape}."
            )

        values = values.astype(
            np.float64,
            copy=False,
        )

        if np.isnan(values).any():
            raise ValueError(
                "Calculated SHAP values contain NaN values"
            )

        if np.isinf(values).any():
            raise ValueError(
                "Calculated SHAP values contain infinite values"
            )

        return values

    def _create_explainer(
        self,
        model: torch.nn.Module,
        background: torch.Tensor,
    ):
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP is not installed. Install it with: "
                "pip install shap"
            ) from exc

        if self.explainer_type == "gradient":
            return shap.GradientExplainer(
                model,
                background,
            )

        if self.explainer_type == "deep":
            return shap.DeepExplainer(
                model,
                background,
            )

        raise ValueError(
            f"Unsupported SHAP explainer: {self.explainer_type}"
        )

    def calculate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> dict:
        """
        Calculate SHAP for one client partition.

        The returned values are additive summaries. They can be
        combined across clients without transmitting raw SHAP rows.
        """

        features = self._collect_features(loader)

        if len(features) < 2:
            raise ValueError(
                "At least two samples are required for SHAP"
            )

        generator = torch.Generator()
        generator.manual_seed(self.random_seed)

        background = self._sample_rows(
            data=features,
            requested_size=self.background_size,
            generator=generator,
        )

        explanation_data = self._sample_rows(
            data=features,
            requested_size=self.explanation_size,
            generator=generator,
        )

        background = background.to(device)
        explanation_data = explanation_data.to(device)

        model = model.to(device)
        model.eval()

        logger.info(
            "Calculating local SHAP | "
            "explainer=%s | available=%d | background=%d | "
            "explained=%d",
            self.explainer_type,
            len(features),
            len(background),
            len(explanation_data),
        )

        explainer = self._create_explainer(
            model=model,
            background=background,
        )

        raw_values = explainer.shap_values(
            explanation_data
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

        # Additive summaries are preferable because they can later
        # be used with distributed or secure aggregation.
        result = {
            "sample_count": int(shap_values.shape[0]),

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