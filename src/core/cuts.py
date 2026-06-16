"""Cut generators for Branch-and-Cut.

Each generator returns a list of (lhs_row, rhs_scalar) satisfying lhs @ x <= rhs.
Cuts must be GLOBALLY valid: they cannot remove any integer-feasible point.
"""
from __future__ import annotations
import math
import itertools
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MIPModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frac(v: float) -> float:
    f = v - math.floor(v)
    return f


def _find_basis(A_full, b_bar, x_aug, m, tol=1e-5):
    """
    Find an m×m invertible basis for the LP at the current optimal solution.

    Searches combinations of size m from (fractional variables) + (zero-valued
    candidates for degenerate basics) until we find an invertible B where
    B @ x_B ≈ b_bar.

    Returns (basic_idx, B_inv, x_B) or (None, None, None) on failure.
    """
    n_aug = A_full.shape[1]

    # Strictly fractional variables are definitely basic
    definite = [
        i for i in range(n_aug)
        if x_aug[i] > tol and not math.isclose(x_aug[i], round(x_aug[i]), abs_tol=tol)
    ]

    # Positive integer-valued variables: may or may not be basic
    positive_int = [
        i for i in range(n_aug)
        if i not in definite and x_aug[i] > tol
    ]

    # Zero-valued: degenerate basic candidates
    zero_vars = [
        i for i in range(n_aug)
        if abs(x_aug[i]) < tol
    ]

    # Build the candidate pool: definite first, then positive_int, then zero_vars
    # We'll try combinations of size m from this pool
    pool = definite + positive_int + zero_vars

    # Cap search to avoid combinatorial explosion
    max_pool = min(len(pool), m + 6)
    pool = pool[:max_pool]

    def _try_basis(indices):
        trial = sorted(indices)
        B = A_full[:, trial]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return None, None, None
        if np.linalg.cond(B) > 1e10:
            return None, None, None
        x_B = B_inv @ b_bar
        # x_B must be non-negative (BFS) and match the actual LP solution
        if (np.all(x_B >= -1e-4) and
                np.allclose(B @ x_B, b_bar, atol=1e-4) and
                np.allclose(x_B, x_aug[trial], atol=1e-4)):
            return trial, B_inv, x_B
        return None, None, None

    for combo in itertools.combinations(pool, m):
        idx, B_inv, x_B = _try_basis(combo)
        if idx is not None:
            return idx, B_inv, x_B

    return None, None, None


# ---------------------------------------------------------------------------
# Gomory fractional cuts (proper implementation via basis reconstruction)
# ---------------------------------------------------------------------------

def gomory_cuts(
    x: np.ndarray,
    model: "MIPModel",
    A_ub_eff: np.ndarray | None,
    b_ub_eff: np.ndarray | None,
    slack: np.ndarray | None = None,
    node_bounds: list | None = None,
    tol: float = 1e-6,
    max_cuts: int = 5,
) -> list[tuple[np.ndarray, float]]:
    """
    Generate Gomory fractional cuts from the LP optimal simplex tableau.

    For each fractional basic integer variable x_k:
      tableau row k: a_bar_kj (for all variables in augmented system)
      Gomory cut: - sum_{j} frac(a_bar_kj) * x_orig_j <= - frac(x_B_k)

    This cut is globally valid (does not remove any integer-feasible point)
    and is violated by the current fractional LP solution.
    """
    n = len(x)

    if A_ub_eff is None or b_ub_eff is None or len(A_ub_eff) == 0:
        return []

    m = len(b_ub_eff)

    # Slack values: s = b - A @ x
    if slack is not None and len(slack) == m:
        s = np.asarray(slack, dtype=float)
    else:
        s = b_ub_eff - A_ub_eff @ x

    # Effective bounds at this node
    eff_bounds = node_bounds if node_bounds is not None else model.bounds
    lb_arr = np.array([b[0] if b[0] is not None else 0.0 for b in eff_bounds])
    ub_arr = np.array([b[1] if b[1] is not None else 1e30 for b in eff_bounds])

    # Augmented system: [A_ub | I_m] @ [x; s] = b_ub
    # BUT for bounded non-basics at UB, the RHS needs adjustment:
    # b_bar = b - sum_{j non-basic at UB} A[:,j] * ub_j
    x_aug = np.concatenate([x, s])
    A_full = np.hstack([A_ub_eff, np.eye(m)])

    # Identify non-basics at upper bound (original variables only)
    nonbasic_UB = [j for j in range(n) if abs(x[j] - ub_arr[j]) < tol and ub_arr[j] < 1e29]

    # Adjust RHS for bounded non-basics at UB
    b_bar = b_ub_eff.copy()
    for j in nonbasic_UB:
        b_bar = b_bar - A_ub_eff[:, j] * ub_arr[j]
    # Also subtract contribution of non-basics at LB (they contribute A[:,j]*lb_j)
    nonbasic_LB = [j for j in range(n) if j not in nonbasic_UB and abs(x[j] - lb_arr[j]) < tol]
    for j in nonbasic_LB:
        if abs(lb_arr[j]) > tol:  # non-zero LB
            b_bar = b_bar - A_ub_eff[:, j] * lb_arr[j]

    # For slacks: they are always at lb=0 when non-basic, so no adjustment needed
    # (slacks at 0 contribute 0 to b_bar)

    # Find the LP basis
    basic_idx, B_inv, x_B = _find_basis(A_full, b_bar, x_aug, m, tol)

    if basic_idx is None:
        return []

    # For bounded variables at UB that are non-basic, shift the variable for Gomory:
    # Let x_j' = ub_j - x_j (so x_j' = 0 when x_j = ub_j). The cut is computed
    # in terms of x_j' for non-basics at UB.
    # For simplicity here: generate cut in terms of original x
    # (the fractional part derivation handles this correctly if we track UB status)

    cuts = []
    int_basic_rows = [
        (k, bi) for k, bi in enumerate(basic_idx)
        if bi < n and model.integrality[bi] > 0
    ]

    for k, bi in int_basic_rows:
        xB_k = x_B[k]
        f_k = _frac(xB_k)
        if f_k < tol or f_k > 1 - tol:
            continue

        # Row k of the full tableau (over all n+m augmented variables)
        tableau_row_k = B_inv[k, :] @ A_full   # shape (n+m,)

        # Gomory cut over ORIGINAL variables (columns 0..n-1)
        # For non-basics at LB: coefficient is -frac(a_kj)
        # For non-basics at UB: coefficient is +frac(-a_kj) = +frac(1-a_kj) if a_kj integer, else -(frac(a_kj))
        # Simplified for LB=0 non-basics:
        # Gomory cut (in <= form):
        #   For non-basic at LB: cut_lhs[j] = -frac(a_kj)
        #   For non-basic at UB: cut_lhs[j] = +frac(-a_kj)   ← sign is POSITIVE
        #   RHS = -frac(b_k) + sum_{j at UB} frac(-a_kj)*ub_j
        # Basic variables contribute 0 since frac(e_k entry)=frac(1)=0.
        cut_lhs = np.zeros(n)
        for j in range(n):
            a_kj = tableau_row_k[j]
            if j in nonbasic_UB:
                cut_lhs[j] = _frac(-a_kj)          # positive contribution
            else:
                cut_lhs[j] = -_frac(a_kj)          # non-positive contribution

        cut_rhs = -f_k
        for j in nonbasic_UB:
            cut_rhs += _frac(-tableau_row_k[j]) * ub_arr[j]

        # Verify the cut violates current LP solution
        violation = cut_lhs @ x - cut_rhs
        if violation > tol:
            cuts.append((cut_lhs, cut_rhs))
            if len(cuts) >= max_cuts:
                break

    return cuts


# ---------------------------------------------------------------------------
# Cover cuts (for knapsack-type binary constraints)
# ---------------------------------------------------------------------------

def cover_cuts(
    x: np.ndarray,
    model: "MIPModel",
    tol: float = 1e-6,
) -> list[tuple[np.ndarray, float]]:
    """
    Generate cover cuts for binary knapsack constraints.

    For a knapsack constraint sum_j a_j x_j <= b (a_j, b >= 0),
    a cover C is a subset with sum_{j in C} a_j > b.
    Cover cut: sum_{j in C} x_j <= |C| - 1.

    This is always valid for 0-1 variables.
    """
    cuts = []
    if model.A_ub is None:
        return cuts

    int_idx = set(model.integer_indices)
    n = len(x)

    for row_idx, row in enumerate(model.A_ub):
        b = model.b_ub[row_idx]

        # Only knapsack-like rows (non-negative coefficients, non-negative RHS)
        if b < -tol or not all(row >= -tol):
            continue

        # Greedy minimal cover: add items in descending weight order
        items = [(j, row[j]) for j in range(n) if row[j] > tol and j in int_idx]
        if not items:
            continue
        items.sort(key=lambda t: t[1], reverse=True)

        cover = []
        total = 0.0
        for j, a in items:
            cover.append(j)
            total += a
            if total > b + tol:
                break

        if total <= b + tol or len(cover) < 2:
            continue

        lhs = np.zeros(n)
        for j in cover:
            lhs[j] = 1.0
        rhs = float(len(cover) - 1)

        if lhs @ x > rhs + tol:
            cuts.append((lhs, rhs))

    return cuts
