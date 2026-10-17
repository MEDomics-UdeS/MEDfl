"""
Federated XGBoost backend for MEDfl real-world mode.

This module adds XGBoost support without modifying the existing PyTorch
real-world client, model, or strategy implementation.
"""

from MEDfl.rw.xgboost.client import XGBoostFlowerClient
from MEDfl.rw.xgboost.strategy import XGBoostStrategy
from MEDfl.rw.xgboost.config import XGBoostConfig

__all__ = [
    "XGBoostFlowerClient",
    "XGBoostStrategy",
    "XGBoostConfig",
]