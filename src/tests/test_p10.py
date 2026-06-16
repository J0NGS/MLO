"""Tests for Problem 10 - Lot Sizing (PLIM - ULSP)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p10_lot_sizing import (
    build_model, decode_solution,
    single_period_warm_start, silver_meal_warm_start,
    sensitivity_analysis, iy, iq, is_,
    DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND, T, PERIOD_NAMES,
)
from core import BranchAndBound, BranchAndCut


def _verify_balance(result, demand, tol=1e-4):
    """Check flow balance s_{t-1}+q_t = d_t+s_t for all t, and s_{T-1}=0."""
    assert result.x is not None
    n_t = len(demand)
    _, q, s = decode_solution(result, n_t)

    # t=0: q_0 = d_0 + s_0  (s_{-1}=0)
    assert abs(q[0] - demand[0] - s[0]) < tol, \
        f"Balance t=0: q={q[0]:.2f} != d+s={demand[0]+s[0]:.2f}"

    for t in range(1, n_t - 1):
        lhs = s[t-1] + q[t]
        rhs = demand[t] + s[t]
        assert abs(lhs - rhs) < tol, \
            f"Balance t={t}: s_prev+q={lhs:.2f} != d+s={rhs:.2f}"

    # t=T-1: s_{T-2}+q_{T-1} = d_{T-1}  (s_{T-1}=0)
    assert abs(s[n_t-2] + q[n_t-1] - demand[n_t-1]) < tol, \
        f"Balance t={n_t-1}: {s[n_t-2]+q[n_t-1]:.2f} != {demand[n_t-1]:.2f}"

    assert abs(s[n_t-1]) < tol, f"Final inventory s_T={s[n_t-1]:.4f} != 0"


def test_original_instance_bb():
    """Default instance: optimal = {P1, P3}, cost = 276."""
    model = build_model(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND, PERIOD_NAMES)
    g_cost, g_x = silver_meal_warm_start(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND)

    result = BranchAndBound(
        model, strategy="best_first", branching="first_fractional",
        initial_incumbent=g_cost, initial_x=g_x, verbose=False,
    ).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 276.0) < 1e-4, \
        f"Expected 276, got {result.obj_value}"

    sp, q, s = decode_solution(result, T)
    assert set(sp) == {0, 2}, \
        f"Expected {{P1,P3}} (indices 0,2), got {{{', '.join(PERIOD_NAMES[t] for t in sp)}}}"

    # Check key inventory values
    assert abs(s[0] - 8.0)  < 1e-4, f"s1={s[0]:.2f} != 8"
    assert abs(s[2] - 17.0) < 1e-4, f"s3={s[2]:.2f} != 17"
    assert abs(s[3] - 12.0) < 1e-4, f"s4={s[3]:.2f} != 12"

    print(f"PASS: bb_original | cost=276 | setup={{P1,P3}} | nodes={result.nodes_explored}")


def test_bb_vs_bc_match():
    """B&B and B&C must find identical cost and setup periods."""
    model = build_model(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND)
    g_cost, g_x = silver_meal_warm_start(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND)

    r_bb = BranchAndBound(model, strategy="best_first", branching="first_fractional",
                          initial_incumbent=g_cost, initial_x=g_x, verbose=False).solve()
    r_bc = BranchAndCut(model, strategy="best_first", branching="first_fractional",
                        cut_types=["gomory"],
                        initial_incumbent=g_cost, initial_x=g_x, verbose=False).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, \
        f"B&B={r_bb.obj_value} != B&C={r_bc.obj_value}"

    sp_bb, _, _ = decode_solution(r_bb, T)
    sp_bc, _, _ = decode_solution(r_bc, T)
    assert set(sp_bb) == set(sp_bc)

    print(f"PASS: bb_vs_bc | cost={r_bb.obj_value:.1f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_warm_starts():
    """Silver-Meal warm-start must be feasible and <= single-period cost."""
    g_sp, x_sp = single_period_warm_start(DEFAULT_SETUP, DEFAULT_DEMAND)
    g_sm, x_sm = silver_meal_warm_start(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND)

    # Single-period: cost = sum(f), always feasible
    assert abs(g_sp - sum(DEFAULT_SETUP)) < 1e-4
    assert g_sm <= g_sp + 1e-6, \
        f"Silver-Meal ({g_sm}) should be <= single-period ({g_sp})"

    # Both must be strictly above the optimal (276)
    model = build_model(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND)
    result = BranchAndBound(model, strategy="best_first", branching="first_fractional",
                            verbose=False).solve()
    assert result.obj_value <= g_sm + 1e-6

    print(f"PASS: warm_starts | single={g_sp:.1f} | SM={g_sm:.1f} | optimal={result.obj_value:.1f}")


def test_balance_constraints():
    """B&B solution must satisfy all flow-balance constraints exactly."""
    model = build_model(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND)
    g_cost, g_x = silver_meal_warm_start(DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND)
    result = BranchAndBound(model, strategy="best_first", branching="first_fractional",
                            initial_incumbent=g_cost, initial_x=g_x, verbose=False).solve()
    assert result.status == "optimal"
    _verify_balance(result, DEFAULT_DEMAND)
    print("PASS: balance | s_{t-1}+q_t = d_t+s_t for all t, s_T=0")


def test_sensitivity_f3():
    """
    Vary setup cost of P3 (index 2):
      f3 <= 130 -> {P1, P3} optimal, cost = f3 + 186
      f3 >= 150 -> {P1, P5} optimal, cost = 326 (constant)
    """
    f3_range = [50.0, 90.0, 130.0, 150.0, 170.0]
    sens = sensitivity_analysis(f3_range, period_idx=2)

    for f3v, total, sp in sens:
        assert total is not None, f"No solution at f3={f3v}"
        _verify_balance_cost(total, f3v, sp)

    print("PASS: sensitivity_f3 | f3<=130: {P1,P3}, f3>=150: {P1,P5}, transition at 140")


def _verify_balance_cost(total, f3v, sp):
    """Helper: check cost and setup periods are consistent."""
    if f3v <= 130.0:
        assert set(sp) == {0, 2}, \
            f"f3={f3v}: expected {{P1,P3}}, got {sp}"
        expected = f3v + 186.0
        assert abs(total - expected) < 1e-3, \
            f"f3={f3v}: expected cost {expected:.1f}, got {total:.1f}"
    elif f3v >= 150.0:
        assert set(sp) == {0, 4}, \
            f"f3={f3v}: expected {{P1,P5}}, got {sp}"
        assert abs(total - 326.0) < 1e-3, \
            f"f3={f3v}: expected cost 326, got {total:.1f}"


if __name__ == "__main__":
    test_original_instance_bb()
    test_bb_vs_bc_match()
    test_warm_starts()
    test_balance_constraints()
    test_sensitivity_f3()
    print("\nAll P10 tests passed!")
