"""Tests for Problem 5 - Production Setup Planning (PLIM)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p5_production_setup import (
    build_model, compute_big_m, sensitivity_analysis,
    DEFAULT_SETUP, DEFAULT_CVAR, DEFAULT_REVENUE,
    DEFAULT_DEMAND, DEFAULT_HOURS, PROD_CAP, HOUR_CAP, CHIPS,
)
from core import BranchAndBound, BranchAndCut


def test_original_instance_bb():
    """B&B must find profit=$520 by producing only chip A (q_A=60, y_A=1)."""
    model = build_model()
    result = BranchAndBound(model, strategy="best_first",
                            branching="most_infeasible").solve()

    assert result.status == "optimal", f"Expected optimal, got {result.status}"
    assert abs(result.obj_value - 520.0) < 1e-4, f"Expected 520, got {result.obj_value}"

    # Only chip A should be active
    n = len(CHIPS)
    y = [round(result.x[n + k]) for k in range(n)]
    assert y[0] == 1, f"Chip A should be produced (y_A={y[0]})"
    assert y[1] == 0, f"Chip B should NOT be produced (y_B={y[1]})"
    assert y[2] == 0, f"Chip C should NOT be produced (y_C={y[2]})"

    q_A = result.x[0]
    assert abs(q_A - 60.0) < 1e-3, f"Expected q_A=60, got {q_A:.4f}"
    print(f"PASS: original_bb | profit={result.obj_value:.2f} | nodes={result.nodes_explored}")


def test_original_instance_bc_matches_bb():
    """B&C must find the same optimal as B&B."""
    model = build_model()

    r_bb = BranchAndBound(model, strategy="best_first",
                          branching="most_infeasible").solve()
    r_bc = BranchAndCut(model, strategy="best_first",
                        branching="most_infeasible",
                        cut_types=["gomory"]).solve()

    assert r_bb.status == "optimal"
    assert r_bc.status == "optimal", f"B&C returned {r_bc.status}"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, (
        f"B&B={r_bb.obj_value:.2f} != B&C={r_bc.obj_value:.2f}"
    )
    print(f"PASS: bb_vs_bc | profit={r_bb.obj_value:.2f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_big_m_tightness():
    """Tight Big-M values must be <= capacity bounds."""
    M = compute_big_m(DEFAULT_HOURS, PROD_CAP, HOUR_CAP)

    for k, chip in enumerate(CHIPS):
        # Big-M should not exceed production cap
        assert M[k] <= PROD_CAP + 1e-9, f"M_{chip}={M[k]} > prod_cap={PROD_CAP}"
        # Big-M should not exceed what hours allow
        from_hours = HOUR_CAP / DEFAULT_HOURS[k]
        assert M[k] <= from_hours + 1e-9, f"M_{chip}={M[k]} > hours_bound={from_hours:.2f}"

    print(f"PASS: big_m_tightness | M = {[f'{m:.2f}' for m in M]}")


def test_sensitivity_switch_point():
    """
    At setup_C <= 150 only chip C is produced; at setup_C >= 250 only A.
    Threshold is near 200 (both give profit=520).
    """
    sens = sensitivity_analysis([100.0, 150.0, 200.0, 250.0, 300.0])
    n = len(CHIPS)

    for sc, profit, x in sens:
        assert profit is not None, f"No solution for setup_C={sc}"
        assert profit > 0, f"Non-positive profit for setup_C={sc}"

        y_A = round(x[n + 0])
        y_C = round(x[n + 2])

        if sc <= 150:
            # Only C should be active (C has more profit than A when setup is cheap)
            assert y_C == 1, f"setup_C={sc}: expected C active, y_C={y_C}"
            assert y_A == 0, f"setup_C={sc}: expected A inactive, y_A={y_A}"
        elif sc >= 250:
            # Only A should be active
            assert y_A == 1, f"setup_C={sc}: expected A active, y_A={y_A}"
            assert y_C == 0, f"setup_C={sc}: expected C inactive, y_C={y_C}"

    print("PASS: sensitivity_switch_point | C-only at sc<=150, A-only at sc>=250")


def test_feasibility_of_solution():
    """The B&B solution must satisfy all production, hours, and Big-M constraints."""
    model = build_model()
    result = BranchAndBound(model, strategy="best_first").solve()

    assert result.status == "optimal"
    x = result.x
    n = len(CHIPS)
    M = compute_big_m(DEFAULT_HOURS, PROD_CAP, HOUR_CAP)

    q = x[:n]
    y = x[n:]

    # Production capacity
    assert sum(q) <= PROD_CAP + 1e-6, f"Production violated: {sum(q):.4f} > {PROD_CAP}"
    # Hours capacity
    hours_used = sum(DEFAULT_HOURS[k] * q[k] for k in range(n))
    assert hours_used <= HOUR_CAP + 1e-6, f"Hours violated: {hours_used:.4f} > {HOUR_CAP}"

    for k in range(n):
        # Big-M: q_k <= M_k * y_k
        assert q[k] <= M[k] * y[k] + 1e-6, (
            f"Big-M violated for chip {CHIPS[k]}: q={q[k]:.4f} > M*y={M[k]*y[k]:.4f}"
        )
        # Min demand: q_k >= dem_k * y_k
        assert q[k] >= DEFAULT_DEMAND[k] * y[k] - 1e-6, (
            f"Min demand violated for {CHIPS[k]}: q={q[k]:.4f} < dem*y={DEFAULT_DEMAND[k]*y[k]:.4f}"
        )
        # Binary y
        assert abs(y[k] - round(y[k])) < 1e-5, f"y_{CHIPS[k]} not binary: {y[k]}"

    print(f"PASS: feasibility | prod={sum(q):.1f}/{PROD_CAP:.0f} | "
          f"hours={hours_used:.1f}/{HOUR_CAP:.0f}")


if __name__ == "__main__":
    test_original_instance_bb()
    test_original_instance_bc_matches_bb()
    test_big_m_tightness()
    test_sensitivity_switch_point()
    test_feasibility_of_solution()
    print("\nAll P5 tests passed!")
