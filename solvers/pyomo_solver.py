# solvers/trivial_glpk.py
"""
Exact extensive-form baseline:
– no learning, no heuristics
– simply feeds the concrete Pyomo model to GLPK
"""

from __future__ import annotations
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import SolverFactory, value

from solvers.base_solver  import BaseSolver
from problems.base_problem import BaseProblem


class PyomoGLPKSolver(BaseSolver):
    """Wraps a vanilla Pyomo→GLPK solve."""

    # short tag used in result filenames / plots
    @property
    def name(self) -> str: 
        return "glpk_extform"

    # nothing to pre-train
    def train(self, problem: BaseProblem, **kwargs) -> None:
        pass

    # one call → one solve
    def solve(
        self, 
        problem : BaseProblem, 
        tee     : bool = False, 
        solver  : str = "glpk",
        **kwargs
    ):
        def _core():
            model = problem.build_model()
            SolverFactory(solver).solve(model, tee=tee)

            # store the exact obj so evaluate() can report mape = 0
            try:
                self.obj_estimate = float(value(model.Obj))
            except (AttributeError, RuntimeError):
                self.obj_estimate = None   # model had no Obj or it’s named differently

            # extract tagged first-stage variable(s)
            fs = [
                v for v in model.component_data_objects(pyo.Var)
                if getattr(v, "is_first_stage", False)
            ]
            x_vals = np.array([float(v.value) for v in fs], dtype=float)
            # return scalar if only one var, else 1-D array
            return float(x_vals[0]) if x_vals.size == 1 else x_vals

        # measure runtime with BaseSolver helper
        return self._timed(_core)
