from typing import Optional

import xgboost as xgb
from flwr.common import Parameters


XGBOOST_TENSOR_TYPE = "xgboost.ubj"


def booster_to_parameters(
    booster: Optional[xgb.Booster],
) -> Parameters:
    if booster is None:
        return Parameters(
            tensors=[],
            tensor_type=XGBOOST_TENSOR_TYPE,
        )

    raw = booster.save_raw(raw_format="ubj")

    return Parameters(
        tensors=[bytes(raw)],
        tensor_type=XGBOOST_TENSOR_TYPE,
    )


def parameters_to_booster(
    parameters: Optional[Parameters],
) -> Optional[xgb.Booster]:
    if parameters is None:
        return None

    if not parameters.tensors:
        return None

    booster = xgb.Booster()
    booster.load_model(bytearray(parameters.tensors[0]))

    return booster


def save_booster(
    booster: xgb.Booster,
    path: str,
) -> None:
    booster.save_model(path)


def load_booster(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster