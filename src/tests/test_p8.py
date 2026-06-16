"""Tests for Problem 8 - CDN Facility Location (PLIM - UFL)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p8_facility_location import (
    build_model, decode_solution, greedy_solution, sensitivity_analysis,
    iy, ix,
    DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS, N_FACILITIES, N_CLIENTS,
    FACILITY_NAMES, CLIENT_NAMES, MAX_OPEN, BUDGET,
)
from core import BranchAndBound, BranchAndCut

# Alias for tests that may reference old name
DEFAULT_LATENCY = DEFAULT_SERVICE_COSTS


def _verify_assignment(result, n_f, n_c, service_costs, fixed_costs, tol=1e-4):
    """Check that every client is assigned to exactly one open facility."""
    assert result.x is not None
    open_f, assignments = decode_solution(result, n_f, n_c)

    for i in range(n_c):
        j = assignments[i]
        assert open_f[j] == 1, (
            f"Client {i} assigned to facility {j} but facility not open"
        )

    total_fixed   = sum(fixed_costs[j] * open_f[j] for j in range(n_f))
    total_service = sum(service_costs[i][assignments[i]] for i in range(n_c))
    expected_cost = total_fixed + total_service
    assert abs(expected_cost - result.obj_value) < tol, (
        f"Decoded cost {expected_cost} != obj_value {result.obj_value}"
    )


def test_original_instance_bb():
    """
    PDF instance: optimal = {C4}, cost = 20 + 33 = 53.
    All clients assigned to C4: R1(9), R2(4), R3(7), R4(2), R5(8), R6(3).
    """
    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS,
                        N_FACILITIES, N_CLIENTS, FACILITY_NAMES, CLIENT_NAMES)
    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)

    result = BranchAndBound(model, strategy="best_first",
                            branching="most_infeasible",
                            initial_incumbent=g_cost, initial_x=g_x).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 53.0) < 1e-4, (
        f"Expected cost 53, got {result.obj_value}"
    )

    open_f, assignments = decode_solution(result, N_FACILITIES, N_CLIENTS)
    _verify_assignment(result, N_FACILITIES, N_CLIENTS, DEFAULT_SERVICE_COSTS, DEFAULT_FIXED_COSTS)

    # C4 (index 3) must be open; others closed
    assert open_f[3] == 1, "C4 should be open"
    assert open_f[0] == 0, "C1 should be closed"
    assert open_f[1] == 0, "C2 should be closed"
    assert open_f[2] == 0, "C3 should be closed"

    print(f"PASS: bb_original | cost=53 | nodes={result.nodes_explored}")


def test_bb_vs_bc_match():
    """B&B and B&C must find identical cost and open the same facilities."""
    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)
    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)

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
    """Greedy (best single-facility) cost must be >= B&B optimal. Greedy = C4 = 53."""
    g_cost, _ = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)

    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)
    result = BranchAndBound(model, strategy="best_first").solve()

    assert result.obj_value <= g_cost + 1e-6, (
        f"Greedy {g_cost} must be >= optimal {result.obj_value}"
    )
    # Best single center is C4: 20 + (9+4+7+2+8+3) = 53
    assert abs(g_cost - 53.0) < 1e-4, f"Expected greedy=53, got {g_cost}"

    print(f"PASS: greedy_upper_bound | greedy={g_cost:.1f} >= optimal={result.obj_value:.1f}")


def test_sensitivity_c4_cost_transition():
    """
    Vary fixed cost of C4:
      - f(C4) = 10: {C4} optimal, cost = 10 + 33 = 43
      - f(C4) = 30: {C2} optimal, cost = 56 (25 + 31)
    Transition at f(C4) = 23: tie ({C4}=56={C2}=56).
    """
    f4_range = [10.0, 30.0]
    sens = sensitivity_analysis(f4_range)

    (f10, cost10, open10), (f30, cost30, open30) = sens

    assert cost10 is not None and cost30 is not None

    # f(C4)=10: C4 optimal
    assert open10[3] == 1, f"f4=10: C4 should be open, got {open10}"
    assert abs(cost10 - 43.0) < 1e-4, f"f4=10: expected 43, got {cost10}"

    # f(C4)=30: C2 optimal
    assert open30[1] == 1, f"f4=30: C2 should be open, got {open30}"
    assert open30[3] == 0, f"f4=30: C4 should be closed, got {open30}"
    assert abs(cost30 - 56.0) < 1e-4, f"f4=30: expected 56, got {cost30}"

    print("PASS: sensitivity_c4 | f4=10: {C4}=43, f4=30: {C2}=56, transition at f4=23")


def test_feasibility_all_constraints():
    """
    Verify that the B&B solution satisfies all UFL constraints:
    - Assignment: sum_j x_{ij} = 1  for all clients i
    - Linking:    x_{ij} <= y_j     for all i, j
    - Budget:     sum_j f_j*y_j <= BUDGET
    - Max open:   sum_j y_j <= MAX_OPEN
    - Min open:   sum_j y_j >= 1
    - All x, y in [0,1]
    """
    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)
    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)
    result = BranchAndBound(model, strategy="best_first",
                            initial_incumbent=g_cost, initial_x=g_x).solve()

    assert result.status == "optimal"
    x = result.x
    n_f, n_c = N_FACILITIES, N_CLIENTS

    # Assignment
    for i in range(n_c):
        total = sum(x[ix(i, j, n_f)] for j in range(n_f))
        assert abs(total - 1.0) < 1e-4, f"Client {i}: sum x_ij = {total:.4f} != 1"

    # Linking and bounds
    open_count = 0
    budget_used = 0.0
    for j in range(n_f):
        y_j = x[iy(j, n_f)]
        assert -1e-6 <= y_j <= 1 + 1e-6, f"y_{j}={y_j} out of [0,1]"
        open_count += round(y_j)
        budget_used += DEFAULT_FIXED_COSTS[j] * round(y_j)
        for i in range(n_c):
            x_ij = x[ix(i, j, n_f)]
            assert x_ij <= y_j + 1e-4, (
                f"Linking violated: x_{i}{j}={x_ij:.4f} > y_{j}={y_j:.4f}"
            )

    # Budget and max-open
    assert budget_used <= BUDGET + 1e-4, f"Budget {budget_used:.1f} exceeds {BUDGET}"
    assert open_count <= MAX_OPEN + 1e-4, f"Opened {open_count} > MAX_OPEN={MAX_OPEN}"
    assert open_count >= 1 - 1e-4, f"Must open at least 1 facility, got {open_count}"

    print("PASS: feasibility | all constraints satisfied "
          f"(open={open_count}, budget={budget_used:.1f})")


if __name__ == "__main__":
    test_original_instance_bb()
    test_bb_vs_bc_match()
    test_greedy_is_upper_bound()
    test_sensitivity_c4_cost_transition()
    test_feasibility_all_constraints()
    print("\nAll P8 tests passed!")
