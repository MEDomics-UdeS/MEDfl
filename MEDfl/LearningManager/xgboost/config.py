from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class XGBoostConfig:
    task: str = "binary"
    mode: str = "bagging"
    local_num_boost_round: int = 10
    params: Dict[str, Any] = field(default_factory=dict)
    num_classes: Optional[int] = None

    def build_params(self) -> Dict[str, Any]:
        if self.task == "binary":
            defaults = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            }

        elif self.task == "regression":
            defaults = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            }

        elif self.task == "multiclass":
            if self.num_classes is None or self.num_classes < 2:
                raise ValueError(
                    "num_classes must be provided and >= 2 "
                    "for multiclass XGBoost."
                )

            defaults = {
                "objective": "multi:softprob",
                "num_class": self.num_classes,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            }

        else:
            raise ValueError(
                f"Unsupported XGBoost task: {self.task}. "
                "Expected binary, regression, or multiclass."
            )

        merged = defaults.copy()
        merged.update(self.params or {})

        return merged