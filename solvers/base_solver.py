# solvers/base_solver.py
"""
Generic interface for learning-augmented *or* exact solvers.

Every concrete solver must inherit BaseSolver and implement:
    • train(connector, **kwargs)   – optional, may be no-op
    • solve(connector, **kwargs)   – must return x⋆  (np.ndarray | float)
    • name                         – short string used in result files
The default evaluate() already computes true objective, MAPE and runtime.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
import time, numpy as np
from problems.base_problem import BaseProblem   # type: ignore


class BaseSolver(ABC):
    # -------- identification ----------------------------------------
    @property
    @abstractmethod
    def name(self) -> str: ...

    # -------- life-cycle hooks --------------------------------------
    @abstractmethod
    def train(self, problem: BaseProblem, **kwargs) -> None:
        """Load data, fit surrogates, etc.  Leave empty if not needed."""
        ...

    @abstractmethod
    def solve(self, problem: BaseProblem, **kwargs) -> Any:
        """
        Must return the first-stage decision (x⋆) as
            • 1-D numpy array     for vector decisions, or
            • float / int         for a scalar decision
        """
        ...

    # -------- evaluation helper (ready-made) ------------------------
    def evaluate(self, problem: BaseProblem, x_star: Any) -> Dict[str, float]:
        """
        Standardised metrics the ExperimentRunner will log.
        Child classes MAY set self.runtime (seconds) before calling.
        """
        true_obj = problem.true_cost(x_star)
        record = {
            "solver": self.name,
            "obj_true": true_obj,
            "runtime": getattr(self, "runtime", None),
        }
        # If the solver stored an *estimated* objective, add MAPE
        est = getattr(self, "obj_estimate", None)
        if est is not None:
            record["obj_pred"] = est
            record["mape"] = abs(est - true_obj) / abs(true_obj) if true_obj else None
        return record

    # -------- convenience timer -------------------------------------
    def _timed(self, fn, *args, **kw):
        start = time.time()
        result = fn(*args, **kw)
        self.runtime = time.time() - start
        return result


