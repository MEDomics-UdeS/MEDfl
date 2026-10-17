from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class XGBoostConfig:
    """
    Configuration object for MEDfl federated XGBoost.

    Parameters
    ----------
    task : str
        One of: "binary", "regression", "multiclass".
    mode : str
        Federated XGBoost mode. First supported mode: "bagging".
    local_num_boost_round : int
        Number of local boosting rounds per federated round.
    params : dict
        Native XGBoost training parameters.
    num_classes : Optional[int]
        Required for multiclass classification.
    """

    task: str = "binary"
    mode: str = "bagging"
    local_num_boost_round: int = 10
    params: Dict[str, Any] = field(default_factory=dict)
    num_classes: Optional[int] = None

    def build_params(self) -> Dict[str, Any]:
        """
        Build a safe default XGBoost parameter dictionary.

        User-provided params override the defaults.
        """

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
                    "num_classes must be provided and >= 2 for multiclass XGBoost."
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
                f"Unsupported XGBoost task '{self.task}'. "
                "Choose from: 'binary', 'regression', 'multiclass'."
            )

        merged = defaults.copy()
        merged.update(self.params or {})
        return merged