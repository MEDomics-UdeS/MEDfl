from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


class XGBoostDataModule:
    """
    Data preparation module for MEDfl federated XGBoost clients.

    This mirrors the behavior of the existing PyTorch real-world client:
      - server-enforced features;
      - server-enforced target;
      - global or per-client split fractions;
      - optional ID-based test split.
    """

    def __init__(
        self,
        data_path: str,
        val_frac: Optional[float] = None,
        test_frac: Optional[float] = None,
        id_col: Optional[str] = None,
        test_ids: Optional[str] = None,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

        self.client_val_frac = val_frac
        self.client_test_frac = test_frac
        self.id_col = id_col
        self.test_ids = test_ids
        self.seed = seed

        self.feature_names = self.df.columns[:-1].tolist()
        self.target_name = self.df.columns[-1]

        self.effective_features = self.feature_names[:]
        self.effective_target = self.target_name
        self.effective_val_frac = 0.0
        self.effective_test_frac = 0.0

    def prepare_from_config(
        self,
        config: Dict[str, Any],
    ) -> Tuple[xgb.DMatrix, Optional[xgb.DMatrix], xgb.DMatrix, Dict[str, Any]]:
        """
        Prepare XGBoost DMatrix objects from server config.
        """

        srv_features = str(config.get("features", "") or "").strip()
        srv_target = str(config.get("target", "") or "").strip()

        if srv_target:
            if srv_target not in self.df.columns:
                raise ValueError(
                    f"Server-specified target '{srv_target}' not found in CSV columns: "
                    f"{list(self.df.columns)}"
                )
            target_col = srv_target
        else:
            target_col = self.df.columns[-1]

        if srv_features:
            feat_cols = [c.strip() for c in srv_features.split(",") if c.strip()]
            missing = [c for c in feat_cols if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Server-specified feature columns not found in CSV: {missing}"
                )
        else:
            feat_cols = [c for c in self.df.columns if c != target_col]

        srv_val = config.get("val_fraction", None)
        srv_test = config.get("test_fraction", None)

        val_frac = (
            float(self.client_val_frac)
            if self.client_val_frac is not None
            else float(srv_val)
            if srv_val is not None
            else 0.10
        )

        test_frac = (
            float(self.client_test_frac)
            if self.client_test_frac is not None
            else float(srv_test)
            if srv_test is not None
            else 0.10
        )

        if not (0.0 <= val_frac < 1.0):
            raise ValueError(f"Invalid val_fraction={val_frac}. Must be in [0, 1).")

        if config.get("id_col") and not self.id_col:
            self.id_col = str(config.get("id_col"))

        if config.get("test_ids") and not self.test_ids:
            self.test_ids = str(config.get("test_ids"))

        X_all = self.df[feat_cols].values
        y_all = self.df[target_col].values

        if self.test_ids and str(self.test_ids).strip():
            X_train, X_val, X_test, y_train, y_val, y_test, actual_test_frac = (
                self._split_with_test_ids(
                    X_all=X_all,
                    y_all=y_all,
                    val_frac=val_frac,
                )
            )
            self.effective_test_frac = actual_test_frac

        else:
            if not (0.0 <= test_frac < 1.0 and val_frac + test_frac < 1.0):
                raise ValueError(
                    f"Invalid fractions: val={val_frac}, test={test_frac}. "
                    "Require val >= 0, test >= 0, val + test < 1."
                )

            X_train, X_val, X_test, y_train, y_val, y_test = self._split_with_fractions(
                X_all=X_all,
                y_all=y_all,
                val_frac=val_frac,
                test_frac=test_frac,
            )
            self.effective_test_frac = test_frac

        self.effective_features = feat_cols
        self.effective_target = target_col
        self.effective_val_frac = val_frac

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)

        dval = None
        if X_val is not None and len(y_val) > 0:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)

        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feat_cols)

        metadata = {
            "features": feat_cols,
            "target": target_col,
            "num_features": len(feat_cols),
            "num_train": int(len(y_train)),
            "num_val": int(len(y_val)) if y_val is not None else 0,
            "num_test": int(len(y_test)),
            "val_fraction": float(self.effective_val_frac),
            "test_fraction": float(self.effective_test_frac),
        }

        return dtrain, dval, dtest, metadata

    def _split_with_fractions(
        self,
        X_all,
        y_all,
        val_frac: float,
        test_frac: float,
    ):
        strat_all = y_all if len(np.unique(y_all)) > 1 else None

        X_trval, X_test, y_trval, y_test = train_test_split(
            X_all,
            y_all,
            test_size=test_frac,
            random_state=self.seed,
            stratify=strat_all,
        )

        if val_frac > 0 and len(y_trval) > 0:
            actual_test_frac = len(y_test) / len(y_all)
            rel_val = val_frac / (1.0 - actual_test_frac)

            strat_tr = y_trval if len(np.unique(y_trval)) > 1 else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_trval,
                y_trval,
                test_size=rel_val,
                random_state=self.seed,
                stratify=strat_tr,
            )
        else:
            X_train, y_train = X_trval, y_trval
            X_val = np.empty((0, X_all.shape[1]))
            y_val = np.empty((0,))

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _split_with_test_ids(
        self,
        X_all,
        y_all,
        val_frac: float,
    ):
        test_ids_list = [i.strip() for i in str(self.test_ids).split(",") if i.strip()]

        if self.id_col and self.id_col in self.df.columns:
            id_series = self.df[self.id_col]

            if np.issubdtype(id_series.dtype, np.number):
                test_ids_list = [int(i) for i in test_ids_list]
            else:
                test_ids_list = [str(i) for i in test_ids_list]

        else:
            id_series = self.df.index

            try:
                test_ids_list = [int(i) for i in test_ids_list]
            except ValueError as exc:
                raise ValueError(
                    "test_ids must be integers when id_col is not provided "
                    "or not found in the CSV."
                ) from exc

        test_mask = id_series.isin(test_ids_list)

        X_test = X_all[test_mask]
        y_test = y_all[test_mask]
        X_trval = X_all[~test_mask]
        y_trval = y_all[~test_mask]

        actual_test_frac = len(y_test) / len(y_all) if len(y_all) > 0 else 0.0

        if val_frac + actual_test_frac >= 1.0:
            raise ValueError(
                f"Validation fraction {val_frac} + actual test fraction "
                f"{actual_test_frac} >= 1."
            )

        if val_frac > 0 and len(y_trval) > 0:
            rel_val = val_frac / (1.0 - actual_test_frac)
            strat_tr = y_trval if len(np.unique(y_trval)) > 1 else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_trval,
                y_trval,
                test_size=rel_val,
                random_state=self.seed,
                stratify=strat_tr,
            )
        else:
            X_train, y_train = X_trval, y_trval
            X_val = np.empty((0, X_all.shape[1]))
            y_val = np.empty((0,))

        return X_train, X_val, X_test, y_train, y_val, y_test, actual_test_frac