"""
MEDfl Strategy wrapper for Flower simulation (Ray)

This file provides a Strategy factory that wraps any Flower server strategy
(e.g., FedAvg, FedProx, FedAdam...) with extra debugging:

✅ What it adds:
- Per-client fit/eval history with round + num_examples (+ loss for eval)
- Per-round server summary history (results/failures, cids, example stats)
- Global-weights fingerprints after aggregation:
    * w_norm, delta_norm, cos_sim_prev, nan_or_inf
- Uses Flower logging (NOT print) so logs show up reliably in simulation
- Optional metrics aggregation functions to remove:
    "No fit_metrics_aggregation_fn provided" warning

Usage (server side):
    strat_factory = Strategy(name="FedAvg", debug_print=True)
    strategy = strat_factory.create_strategy()

    fl.simulation.start_simulation(..., strategy=strategy, ...)

After run:
    strategy.round_history
    strategy.client_fit_history
    strategy.client_eval_history
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, List, Tuple, Callable

import numpy as np
import flwr as fl
import optuna
import logging


Metrics = Dict[str, float]
MetricsAggFn = Callable[[List[Tuple[int, Metrics]]], Metrics]


def weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Weighted average of client metrics by num_examples.

    Flower passes:
        metrics = [(num_examples, {"rmse": 1.2, "mae": 0.5}), ...]
    """
    if not metrics:
        return {}

    total = sum(int(n) for n, _ in metrics)
    if total <= 0:
        return {}

    # union of all keys
    keys = set()
    for _, m in metrics:
        keys |= set(m.keys())

    out: Dict[str, float] = {}
    for k in keys:
        acc = 0.0
        used = 0
        for n, m in metrics:
            if k not in m:
                continue
            try:
                v = float(m[k])
            except Exception:
                continue
            acc += float(n) * v
            used += int(n)
        if used > 0:
            out[k] = acc / float(total)

    return out


class Strategy:
    def __init__(
        self,
        name: str = "FedAvg",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters=None,
        evaluation_methode: str = "centralized",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        debug_print: bool = True,
        enable_metrics_aggregation: bool = True,  # ✅ remove "No fit_metrics_aggregation_fn" warnings
    ) -> None:
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters if initial_parameters is not None else []
        self.evaluate_fn = None
        self.name = name
        self.evaluation_methode = evaluation_methode
        self.strategy_kwargs = strategy_kwargs or {}
        self.debug_print = debug_print
        self.enable_metrics_aggregation = enable_metrics_aggregation

        # Exposed after create_strategy()
        self.strategy_object: Optional[fl.server.strategy.Strategy] = None

        # Optuna
        self.study: Optional[optuna.Study] = None
        self.hpo_rate: Optional[int] = None
        self.params_config: Optional[dict] = None

    def optuna_fed_optimization(self, direction: str, hpo_rate: int, params_config):
        self.study = optuna.create_study(direction=direction)
        self.hpo_rate = hpo_rate
        self.params_config = params_config

    def get_strategy_by_name(self):
        try:
            return getattr(fl.server.strategy, self.name)
        except AttributeError as e:
            available = [n for n in dir(fl.server.strategy) if n and n[0].isupper()]
            raise ValueError(f"Unknown strategy '{self.name}'. Available: {available}") from e

    def create_strategy(self):
        BaseStrategy: Type[fl.server.strategy.Strategy] = self.get_strategy_by_name()

        debug_print = self.debug_print

        class TrackingStrategy(BaseStrategy):  # type: ignore[misc]
            """
            Wraps a Flower Strategy and adds:
            - per-client fit/eval history
            - per-round summary history
            - global weights fingerprints after aggregation
            - robust logging (shows in simulation logs)
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # cid -> list of dicts per round
                self.client_fit_history: Dict[str, list] = {}
                self.client_eval_history: Dict[str, list] = {}

                # list of dicts (round summaries)
                self.round_history: list = []

                # previous global model (flattened) for delta/cos sim
                self._prev_global_flat: Optional[np.ndarray] = None

                self.latest_aggregated_parameters = None

                # logger (Flower config will handle output)
                self._log = logging.getLogger("flwr")
                self._debug_print = debug_print

                # prove that THIS strategy is active
                if self._debug_print:
                    self._log.warning("✅ TrackingStrategy is ACTIVE (debug enabled)")

            # ---------- helpers ----------
            def _params_to_flat(self, parameters: fl.common.Parameters) -> np.ndarray:
                nds = fl.common.parameters_to_ndarrays(parameters)
                if not nds:
                    return np.array([], dtype=np.float64)
                return np.concatenate([x.reshape(-1).astype(np.float64, copy=False) for x in nds])

            def _fingerprint(self, flat: np.ndarray) -> Dict[str, Any]:
                if flat.size == 0:
                    return {
                        "w_norm": 0.0,
                        "nan_or_inf": False,
                        "delta_norm": None,
                        "cos_sim_prev": None,
                    }

                w_norm = float(np.linalg.norm(flat))
                nan_or_inf = bool(np.isnan(flat).any() or np.isinf(flat).any())

                delta_norm = None
                cos_sim = None
                if self._prev_global_flat is not None and self._prev_global_flat.size == flat.size:
                    diff = flat - self._prev_global_flat
                    delta_norm = float(np.linalg.norm(diff))

                    denom = np.linalg.norm(flat) * np.linalg.norm(self._prev_global_flat)
                    if denom > 0:
                        cos_sim = float(np.dot(flat, self._prev_global_flat) / denom)

                return {
                    "w_norm": w_norm,
                    "nan_or_inf": nan_or_inf,
                    "delta_norm": delta_norm,
                    "cos_sim_prev": cos_sim,
                }

            def _safe_failure_summaries(self, failures: list, cap: int = 5) -> List[str]:
                return [str(f) for f in failures][:cap]

            # ---------- overrides ----------
            def aggregate_fit(self, server_round, results, failures):
                # Store per-client fit history WITH round + num_examples
                cids: List[str] = []
                num_examples_list: List[int] = []

                for client_proxy, fit_res in results:
                    cid = getattr(client_proxy, "cid", "unknown")
                    cids.append(cid)

                    ne = int(getattr(fit_res, "num_examples", 0))
                    num_examples_list.append(ne)

                    entry = {
                        "round": int(server_round),
                        "num_examples": ne,
                        **dict(fit_res.metrics or {}),
                    }
                    # store client weights only for regression runs
                    if fit_res.metrics is not None and "train_rmse" in fit_res.metrics:
                        client_nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
                        entry["weights"] = [arr.copy() for arr in client_nds]
                    self.client_fit_history.setdefault(cid, []).append(entry)

                # Call base aggregation
                agg = super().aggregate_fit(server_round, results, failures)
        

                # Round summary + global fingerprint (only if aggregation produced parameters)
                if agg is not None:
                    parameters_agg, metrics_agg = agg  # type: ignore[misc]
                    self.latest_aggregated_parameters = (parameters_agg)
                    flat = self._params_to_flat(parameters_agg)
                    fp = self._fingerprint(flat)
                    self._prev_global_flat = flat

                    nds = fl.common.parameters_to_ndarrays(parameters_agg)

                    round_row = {
                        "round": int(server_round),
                        "phase": "fit",
                        "n_results": int(len(results)),
                        "n_failures": int(len(failures)),
                        "cids": cids,
                        "total_examples": int(sum(num_examples_list)),
                        "min_examples": int(min(num_examples_list)) if num_examples_list else 0,
                        "max_examples": int(max(num_examples_list)) if num_examples_list else 0,
                        "failures": self._safe_failure_summaries(failures),
                        "weights": [arr.copy() for arr in nds],   # <- add this
                        **fp,
                        **(dict(metrics_agg or {})),
                    }
                    
                    self.round_history.append(round_row)

                    if self._debug_print:
                        self._log.warning(
                            f"[fit r={server_round}] "
                            f"results={round_row['n_results']} failures={round_row['n_failures']} "
                            f"total_ex={round_row['total_examples']} "
                            f"w_norm={round_row['w_norm']:.4g} "
                            f"delta={round_row['delta_norm']} "
                            f"cos={round_row['cos_sim_prev']} "
                            f"nan/inf={round_row['nan_or_inf']}"
                        )

                return agg

            def aggregate_evaluate(self, server_round, results, failures):
                cids: List[str] = []
                num_examples_list: List[int] = []

                for client_proxy, eval_res in results:
                    cid = getattr(client_proxy, "cid", "unknown")
                    cids.append(cid)

                    ne = int(getattr(eval_res, "num_examples", 0))
                    num_examples_list.append(ne)

                    entry = {
                        "round": int(server_round),
                        "num_examples": ne,
                        "loss": float(getattr(eval_res, "loss", float("nan"))),
                        **dict(eval_res.metrics or {}),
                    }
                    self.client_eval_history.setdefault(cid, []).append(entry)

                agg = super().aggregate_evaluate(server_round, results, failures)

                if agg is not None:
                    loss_agg, metrics_agg = agg  # type: ignore[misc]
                    round_row = {
                        "round": int(server_round),
                        "phase": "eval",
                        "n_results": int(len(results)),
                        "n_failures": int(len(failures)),
                        "cids": cids,
                        "total_examples": int(sum(num_examples_list)),
                        "min_examples": int(min(num_examples_list)) if num_examples_list else 0,
                        "max_examples": int(max(num_examples_list)) if num_examples_list else 0,
                        "failures": self._safe_failure_summaries(failures),
                        "loss": float(loss_agg) if loss_agg is not None else None,
                        **(dict(metrics_agg or {})),
                    }
                    self.round_history.append(round_row)

                    if self._debug_print:
                        self._log.warning(
                            f"[eval r={server_round}] "
                            f"results={round_row['n_results']} failures={round_row['n_failures']} "
                            f"loss={round_row['loss']}"
                        )

                return agg

            def get_debug_state(self) -> Dict[str, Any]:
                return {
                    "round_history": self.round_history,
                    "client_fit_history": self.client_fit_history,
                    "client_eval_history": self.client_eval_history,
                }

        # ---- base kwargs ----
        kwargs: Dict[str, Any] = dict(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_available_clients,
        )

        # Initial parameters
        if self.initial_parameters is not None:
            kwargs["initial_parameters"] = fl.common.ndarrays_to_parameters(self.initial_parameters)

        # Optional evaluate function
        if self.evaluate_fn is not None:
            kwargs["evaluate_fn"] = self.evaluate_fn

        # ✅ Metrics aggregation (removes warnings and gives aggregated metrics)
        if self.enable_metrics_aggregation:
            kwargs["fit_metrics_aggregation_fn"] = weighted_avg
            kwargs["evaluate_metrics_aggregation_fn"] = weighted_avg

        # User strategy overrides
        kwargs.update(self.strategy_kwargs)

        self.strategy_object = TrackingStrategy(**kwargs)
        return self.strategy_object