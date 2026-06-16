"""Tests for Problem 8 - CDN Facility Location (PLIM - UFL)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p8_facility_location import (
    build_model, decode_solution, greedy_solution, sensitivity_analysis,
    iy, ix,
    DEFAULT_FIXED_COSTS, DEFAULT_LATENCY, N_FACILITIES, N_CLIENTS,
    FACILITY_NAMES, CLIENT_NAMES,
)
from core import BranchAndBound, BranchAndCut


def _verify_assignment(result, n_f, n_c, latency, fixed_costs, tol=1e-4):
    """Check that every client is assigned to exactly one open facility."""
    assert result.x is not None
    open_f, assignments = decode_solution(result, n_f, n_c)

    # Every facility used must be open
    for i in range(n_c):
        j = assignments[i]
        assert open_f[j] == 1, (
            f"Client {i} assigned to facility {j} but facility not open"
        )

    # Recompute cost
    total_fixed   = sum(fixed_costs[j] * open_f[j] for j in range(n_f))
    total_latency = sum(latency[i][assignments[i]] for i in range(n_c))
    expected_cost = total_fixed + total_latency
    assert abs(expected_cost - result.obj_value) < tol, (
        f"Decoded cost {expected_cost} != obj_value {result.obj_value}"
    )


def test_original_instance_bb():
    """
    Default instance: optimal = {S1, S2}, cost = 90 + 63 = 153.
    C1->S1(5), C2->S2(5), C3->S2(45), C4->S1(8).
    """
    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY,
                        N_FACILITIES, N_CLIENTS, FACILITY_NAMES, CLIENT_NAMES)
    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)

    result = BranchAndBound(model, strategy="best_first",
                            branching="most_infeasible",
                            initial_incumbent=g_cost, initial_x=g_x).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 153.0) < 1e-4, (
        f"Expected cost 153, got {result.obj_value}"
    )

    open_f, assignments = decode_solution(result, N_FACILITIES, N_CLIENTS)
    _verify_assignment(result, N_FACILITIES, N_CLIENTS, DEFAULT_LATENCY, DEFAULT_FIXED_COSTS)

    # S1 and S2 must be open; S3 must be closed
    assert open_f[0] == 1, "S1 should be open"
    assert open_f[1] == 1, "S2 should be open"
    assert open_f[2] == 0, "S3 should be closed"

    print(f"PASS: bb_original | cost=153 | nodes={result.nodes_explored}")


def test_bb_vs_bc_match():
    """B&B and B&C must find identical cost and open the same facilities."""
    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)
    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)

    r_bb = BranchAndBound(model, strategy="best_first",
                          initial_incumbent=g_cost, initial_x=g_x).solve()
    r_bc = BranchAndCut(model, strategy="best_first", cut_types=["gomory"],
                        initial_incumbent=g_cost, initial_x=g_x).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, (
        f"B&B={r_bb.obj_value} != B&C={r_bc.obj_value}"
    )

    open_bb, _ = decode_solution(r_bb, N_FACILITIES, N_CLIENTS)
    open_bc, _ = decode_solution(r_bc, N_FACILITIES, N_CLIENTS)
    assert open_bb == open_bc, (
        f"B&B opens {open_bb} but B&C opens {open_bc}"
    )

    print(f"PASS: bb_vs_bc | cost={r_bb.obj_value:.1f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_greedy_is_upper_bound():
    """Greedy (best single-facility) cost must be >= B&B optimal."""
    g_cost, _ = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)

    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)
    result = BranchAndBound(model, strategy="best_first").solve()

    assert result.obj_value <= g_cost + 1e-6, (
        f"Greedy {g_cost} must be >= optimal {result.obj_value}"
    )
    # Best single server is S2 alone: 30 + 50+5+45+48 = 178
    assert abs(g_cost - 178.0) < 1e-4, f"Expected greedy=178, got {g_cost}"

    print(f"PASS: greedy_upper_bound | greedy={g_cost:.1f} >= optimal={result.obj_value:.1f}")


def test_sensitivity_s3_cost_transition():
    """
    Vary fixed cost of S3:
      - S3 cost = 20..50: {S2, S3} optimal, cost = S3_cost + 98
      - S3 cost = 60..80: {S1, S2} optimal, cost = 153 (constant)
    Transition: at S3_cost = 55, {S2,S3} cost = 153 = {S1,S2} cost.
    """
    f3_range = [20.0, 40.0, 50.0, 60.0, 80.0]
    sens = sensitivity_analysis(f3_range)

    for f3, total, open_f in sens:
        assert total is not None, f"No solution at S3_cost={f3}"
        assert open_f is not None

        if f3 <= 50.0:
            # S2+S3 optimal
            assert open_f[0] == 0, f"S3={f3}: S1 should be closed, got open_f={open_f}"
            assert open_f[1] == 1, f"S3={f3}: S2 should be open"
            assert open_f[2] == 1, f"S3={f3}: S3 should be open"
            expected = f3 + 98.0
            assert abs(total - expected) < 1e-4, (
                f"S3={f3}: expected {expected}, got {total}"
            )
        elif f3 >= 60.0:
            # S1+S2 optimal
            assert open_f[0] == 1, f"S3={f3}: S1 should be open"
            assert open_f[1] == 1, f"S3={f3}: S2 should be open"
            assert open_f[2] == 0, f"S3={f3}: S3 should be closed"
            assert abs(total - 153.0) < 1e-4, (
                f"S3={f3}: expected 153, got {total}"
            )

    print("PASS: sensitivity_s3 | transition at S3_cost=55: {S2,S3}<->{S1,S2}")


def test_feasibility_all_constraints():
    """
    Verify that the B&B solution satisfies all UFL constraints:
    - Assignment: sum_j x_{ij} = 1  for all clients i
    - Linking:    x_{ij} <= y_j     for all i, j
    - All x, y in [0,1]
    """
    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)
    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)
    result = BranchAndBound(model, strategy="best_first",
                            initial_incumbent=g_cost, initial_x=g_x).solve()

    assert result.status == "optimal"
    x = result.x
    n_f, n_c = N_FACILITIES, N_CLIENTS

    for i in range(n_c):
        total = sum(x[ix(i, j, n_f)] for j in range(n_f))
        assert abs(total - 1.0) < 1e-4, (
            f"Client {i}: sum x_ij = {total:.4f} != 1"
        )

    for j in range(n_f):
        y_j = x[iy(j, n_f)]
        assert -1e-6 <= y_j <= 1 + 1e-6, f"y_{j}={y_j} out of [0,1]"
        for i in range(n_c):
            x_ij = x[ix(i, j, n_f)]
            assert x_ij <= y_j + 1e-4, (
                f"Linking violated: x_{i}{j}={x_ij:.4f} > y_{j}={y_j:.4f}"
            )

    print("PASS: feasibility | all assignment and linking constraints satisfied")


if __name__ == "__main__":
    test_original_instance_bb()
    test_bb_vs_bc_match()
    test_greedy_is_upper_bound()
    test_sensitivity_s3_cost_transition()
    test_feasibility_all_constraints()
    print("\nAll P8 tests passed!")
