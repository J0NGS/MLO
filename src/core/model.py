from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MIPModel:
    """Generic Mixed-Integer Program in standard form.

    Solves: sense{c @ x}
    Subject to:
        A_ub @ x <= b_ub   (optional)
        A_eq @ x == b_eq   (optional)
        bounds[i] = (lb_i, ub_i) for each variable
        integrality[i] = 0 (continuous), 1 (integer), 2 (binary)
    """
    c: np.ndarray
    bounds: list[tuple[float | None, float | None]]
    integrality: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    sense: str = "min"          # "min" or "max"
    var_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=float)
        self.integrality = np.asarray(self.integrality, dtype=int)
        if self.A_ub is not None:
            self.A_ub = np.asarray(self.A_ub, dtype=float)
            self.b_ub = np.asarray(self.b_ub, dtype=float)
        if self.A_eq is not None:
            self.A_eq = np.asarray(self.A_eq, dtype=float)
            self.b_eq = np.asarray(self.b_eq, dtype=float)
        if not self.var_names:
            self.var_names = [f"x{i}" for i in range(len(self.c))]

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def integer_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.integrality) if v > 0]

    def effective_c(self) -> np.ndarray:
        """Returns c adjusted for maximization (linprog always minimizes)."""
        return self.c if self.sense == "min" else -self.c
