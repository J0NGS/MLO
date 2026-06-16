"""Tests for Problem 9 - Bin Packing (PIB)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p9_bin_packing import (
    build_model, decode_solution, ffd_assignment, ffd_to_x_vector,
    sensitivity_analysis, iy, ix,
    DEFAULT_WEIGHTS, DEFAULT_CAPACITY, N_ITEMS, N_BINS,
    ITEM_NAMES, BIN_NAMES,
)
from core import BranchAndBound, BranchAndCut


def _verify_packing(result, weights, capacity, n_bins, tol=1e-4):
    """Check that each item is assigned to exactly one bin and loads are within capacity."""
    assert result.x is not None
    n_items = len(weights)
    n_open, bin_contents = decode_solution(result, n_items, n_bins)

    # Each item appears in exactly one bin
    item_count = {i: 0 for i in range(n_items)}
    for j, items in bin_contents.items():
        load = sum(weights[i] for i in items)
        assert load <= capacity + tol, (
            f"Bin {j} overloaded: {load:.1f} > {capacity:.1f}"
        )
        for i in items:
            item_count[i] += 1

    for i in range(n_items):
        assert item_count[i] == 1, (
            f"Item {i} appears in {item_count[i]} bins (expected 1)"
        )


def test_original_instance_bb():
    """Default instance: B=10, weights=[6,4,4,3,3,2]. Optimal = 3 bins."""
    model = build_model(DEFAULT_WEIGHTS, DEFAULT_CAPACITY, N_BINS, ITEM_NAMES, BIN_NAMES)
    item_bin, ffd_n = ffd_assignment(DEFAULT_WEIGHTS, DEFAULT_CAPACITY)
    g_x = ffd_to_x_vector(item_bin, N_ITEMS, N_BINS)

    result = BranchAndBound(
        model, strategy="best_first", branching="first_fractional",
        initial_incumbent=float(ffd_n), initial_x=g_x, verbose=False,
    ).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 3.0) < 1e-4, (
        f"Expected 3 bins, got {result.obj_value}"
    )
    _verify_packing(result, DEFAULT_WEIGHTS, DEFAULT_CAPACITY, N_BINS)

    print(f"PASS: bb_original | bins=3 | nodes={result.nodes_explored}")


def test_bb_vs_bc_match():
    """B&B and B&C must find the same optimal number of bins."""
    model = build_model(DEFAULT_WEIGHTS, DEFAULT_CAPACITY, N_BINS)
    item_bin, ffd_n = ffd_assignment(DEFAULT_WEIGHTS, DEFAULT_CAPACITY)
    g_x = ffd_to_x_vector(item_bin, N_ITEMS, N_BINS)

    r_bb = BranchAndBound(
        model, strategy="best_first", branching="first_fractional",
        initial_incumbent=float(ffd_n), initial_x=g_x, verbose=False,
    ).solve()
    r_bc = BranchAndCut(
        model, strategy="best_first", branching="first_fractional",
        cut_types=["gomory"],
        initial_incumbent=float(ffd_n), initial_x=g_x, verbose=False,
    ).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, (
        f"B&B={r_bb.obj_value} != B&C={r_bc.obj_value}"
    )
    _verify_packing(r_bb, DEFAULT_WEIGHTS, DEFAULT_CAPACITY, N_BINS)
    _verify_packing(r_bc, DEFAULT_WEIGHTS, DEFAULT_CAPACITY, N_BINS)

    print(f"PASS: bb_vs_bc | bins={r_bb.obj_value:.0f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_ffd_is_upper_bound():
    """FFD solution must be feasible and its cost >= B&B optimal."""
    item_bin, ffd_n = ffd_assignment(DEFAULT_WEIGHTS, DEFAULT_CAPACITY)

    # Check FFD feasibility
    bin_loads: dict[int, float] = {}
    for i, j in item_bin.items():
        bin_loads[j] = bin_loads.get(j, 0.0) + DEFAULT_WEIGHTS[i]
    for j, load in bin_loads.items():
        assert load <= DEFAULT_CAPACITY + 1e-6, f"FFD bin {j} overloaded: {load}"

    model = build_model(DEFAULT_WEIGHTS, DEFAULT_CAPACITY, N_BINS)
    result = BranchAndBound(
        model, strategy="best_first", branching="first_fractional", verbose=False,
    ).solve()

    assert result.obj_value <= ffd_n + 1e-6, (
        f"Optimal {result.obj_value} must be <= FFD {ffd_n}"
    )
    # For this instance FFD is optimal at B=10
    assert abs(ffd_n - 3.0) < 1e-4
    print(f"PASS: ffd_upper_bound | FFD={ffd_n} >= optimal={result.obj_value:.0f}")


def test_ffd_suboptimal_at_b11():
    """
    At capacity B=11, FFD gives 3 bins but optimal is 2.
    Optimal: {I1,I4,I6}={6,3,2}=11, {I2,I3,I5}={4,4,3}=11.
    """
    B = 11.0
    W = DEFAULT_WEIGHTS
    n = N_ITEMS

    model = build_model(W, B, n)
    item_bin, ffd_n = ffd_assignment(W, B)
    g_x = ffd_to_x_vector(item_bin, n, n)

    assert ffd_n == 3, f"Expected FFD=3 at B=11, got {ffd_n}"

    result = BranchAndBound(
        model, strategy="best_first", branching="first_fractional",
        initial_incumbent=float(ffd_n), initial_x=g_x, verbose=False,
    ).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 2.0) < 1e-4, (
        f"Expected optimal=2 at B=11, got {result.obj_value}"
    )
    _verify_packing(result, W, B, n)
    print(f"PASS: ffd_suboptimal_b11 | FFD={ffd_n} > optimal=2 | nodes={result.nodes_explored}")


def test_sensitivity_capacity():
    """
    Vary capacity B from 6 to 13:
      B=6,7   -> 4 bins (sum=22 > 3*7=21)
      B=8..10 -> 3 bins
      B=11..13-> 2 bins (transition where FFD is suboptimal at B=11)
    """
    cap_range = [float(v) for v in range(6, 14)]
    sens = sensitivity_analysis(cap_range, DEFAULT_WEIGHTS)

    expected = {
        6.0: 4.0, 7.0: 4.0,
        8.0: 3.0, 9.0: 3.0, 10.0: 3.0,
        11.0: 2.0, 12.0: 2.0, 13.0: 2.0,
    }
    for B_val, n_opt, ffd_bins in sens:
        assert n_opt is not None, f"No solution at B={B_val}"
        assert abs(n_opt - expected[B_val]) < 1e-4, (
            f"B={B_val}: expected {expected[B_val]} bins, got {n_opt}"
        )

    # B=11: FFD must be suboptimal
    b11 = next((row for row in sens if row[0] == 11.0), None)
    assert b11 is not None
    _, opt11, ffd11 = b11
    assert ffd11 == 3 and abs(opt11 - 2.0) < 1e-4, (
        f"B=11: expected FFD=3 > optimal=2, got FFD={ffd11}, opt={opt11}"
    )

    print("PASS: sensitivity_capacity | B=6,7->4 | B=8-10->3 | B=11-13->2 (FFD suboptimal at 11)")


if __name__ == "__main__":
    test_original_instance_bb()
    test_bb_vs_bc_match()
    test_ffd_is_upper_bound()
    test_ffd_suboptimal_at_b11()
    test_sensitivity_capacity()
    print("\nAll P9 tests passed!")
