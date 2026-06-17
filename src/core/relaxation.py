from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, OptimizeResult
from dataclasses import dataclass

from .model import MIPModel


@dataclass
class LPResult:
    status: str          # "optimal", "infeasible", "unbounded", "error"
    obj_value: float | None
    x: np.ndarray | None
    slack: np.ndarray | None = None   # b_ub - A_ub @ x  (for Gomory cuts)
    A_ub_used: np.ndarray | None = None
    b_ub_used: np.ndarray | None = None


def solve_relaxation(
    model: MIPModel,
    extra_ub_lhs: np.ndarray | None = None,
    extra_ub_rhs: np.ndarray | None = None,
    node_bounds: list[tuple[float | None, float | None]] | None = None,
) -> LPResult:
    """Solve LP relaxation of model, optionally with extra cuts and tightened bounds."""

    # em problemas de max, o model.effective_c() inverte o sinal.
    c = model.effective_c()

    # monta A_ub/b_ub efetivos combinando as restrições originais do modelo
    # com cortes adicionais (se houver) vindos do branch-and-cut.
    A_ub_parts, b_ub_parts = [], []
    if model.A_ub is not None:
        A_ub_parts.append(model.A_ub)
        b_ub_parts.append(model.b_ub)
    if extra_ub_lhs is not None and len(extra_ub_lhs) > 0:
        A_ub_parts.append(np.atleast_2d(extra_ub_lhs))
        b_ub_parts.append(np.atleast_1d(extra_ub_rhs))

    A_ub = np.vstack(A_ub_parts) if A_ub_parts else None
    b_ub = np.concatenate(b_ub_parts) if b_ub_parts else None

    # limites ativos da variável no nó atual; se não houver bounds do nó,
    # usa os bounds globais definidos no modelo.
    bounds = node_bounds if node_bounds is not None else model.bounds
    ## ----------------------------------------------------------
    # resolve a relaxação LP
    result: OptimizeResult = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=bounds,
        method="highs",
    )
    ## ----------------------------------------------------------

    # mapeamento dos status do HiGHS para o contrato interno LPResult:
    # 0 = ótimo, 2 = inviável, 3 = ilimitado.
    if result.status == 0:
        # result.fun está no espaço transformado (min). para modelos de max,
        # convertemos de volta para o valor objetivo no sentido original.
        raw_obj = float(result.fun)
        obj = raw_obj if model.sense == "min" else -raw_obj
        slack = result.slack if hasattr(result, "slack") else None
        return LPResult("optimal", obj, result.x, slack, A_ub, b_ub)
    elif result.status == 2:
        return LPResult("infeasible", None, None)
    elif result.status == 3:
        return LPResult("unbounded", None, None)
    else:
        return LPResult("error", None, None)
