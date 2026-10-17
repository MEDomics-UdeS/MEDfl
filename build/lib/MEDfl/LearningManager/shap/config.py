from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class SHAPConfig:
    """
    Configuration for the optional post-training federated SHAP phase.
    """

    enabled: bool = False

    # Neural networks:
    #   gradient
    #   deep
    #
    # XGBoost:
    #   tree
    explainer: Literal[
        "gradient",
        "deep",
        "tree",
    ] = "gradient"

    data_split: Literal[
        "train",
        "validation",
        "test",
    ] = "validation"

    background_size: int = 100
    explanation_size: int = 500
    random_seed: int = 42

    feature_names: Optional[List[str]] = None

    minimum_samples: int = 10
    clipping_value: Optional[float] = None

    include_client_results: bool = True

    # TreeExplainer output:
    #
    # raw:
    #   Binary classification → log-odds margin
    #   Regression → raw prediction
    #
    # probability:
    #   Binary classification probability when supported by the
    #   installed SHAP/XGBoost versions.
    model_output: Literal[
        "raw",
        "probability",
    ] = "raw"

    def validate(
        self,
        input_size: int,
    ) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "SHAP enabled must be a boolean"
            )

        if self.explainer not in {
            "gradient",
            "deep",
            "tree",
        }:
            raise ValueError(
                "SHAP explainer must be 'gradient', "
                "'deep', or 'tree'"
            )

        if self.data_split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                "SHAP data_split must be train, "
                "validation, or test"
            )

        if self.background_size <= 0:
            raise ValueError(
                "SHAP background_size must be greater than zero"
            )

        if self.explanation_size <= 0:
            raise ValueError(
                "SHAP explanation_size must be greater than zero"
            )

        if self.minimum_samples <= 0:
            raise ValueError(
                "SHAP minimum_samples must be greater than zero"
            )

        if self.clipping_value is not None:
            if self.clipping_value <= 0:
                raise ValueError(
                    "SHAP clipping_value must be greater than zero"
                )

        if self.model_output not in {
            "raw",
            "probability",
        }:
            raise ValueError(
                "SHAP model_output must be 'raw' or "
                "'probability'"
            )

        if self.feature_names is not None:
            if len(self.feature_names) != input_size:
                raise ValueError(
                    "The number of SHAP feature names must "
                    "match the model input size. "
                    f"Expected {input_size}, "
                    f"received {len(self.feature_names)}."
                )

            if len(set(self.feature_names)) != len(
                self.feature_names
            ):
                raise ValueError(
                    "SHAP feature names must be unique"
                )

    def validate_backend(
        self,
        input_size: int,
        backend: str,
        task: Optional[str] = None,
    ) -> None:
        """
        Validate both the general configuration and the selected
        machine-learning backend.
        """

        self.validate(input_size)

        backend = str(backend).lower()

        if backend == "pytorch":
            if self.explainer not in {
                "gradient",
                "deep",
            }:
                raise ValueError(
                    "PyTorch models require explainer='gradient' "
                    "or explainer='deep'"
                )

        elif backend == "xgboost":
            if self.explainer != "tree":
                raise ValueError(
                    "XGBoost models require explainer='tree'"
                )

            if task == "multiclass":
                raise ValueError(
                    "The current federated XGBoost SHAP "
                    "implementation supports binary classification "
                    "and regression. Multiclass requires "
                    "class-specific SHAP aggregation."
                )

            if (
                task == "regression"
                and self.model_output == "probability"
            ):
                raise ValueError(
                    "model_output='probability' is not valid for "
                    "XGBoost regression"
                )

        else:
            raise ValueError(
                f"Unsupported SHAP backend: {backend}"
            )

    def resolved_feature_names(
        self,
        input_size: int,
    ) -> List[str]:
        if self.feature_names is not None:
            return list(self.feature_names)

        return [
            f"feature_{index}"
            for index in range(input_size)
        ]