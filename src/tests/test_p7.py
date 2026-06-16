"""Tests for Problem 7 - TSP (PIB - DFJ formulation)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p7_tsp import (
    build_model, decode_tour, nearest_neighbor_tour, tour_to_x_vector,
    sensitivity_analysis, tour_cost,
    DEFAULT_DIST, N, CITY_NAMES,
    TSPBranchAndCut,
)
from core import BranchAndBound


def _verify_tour(tour, n, dist):
    """Tour must visit all n cities exactly once and return to start."""
    assert tour is not None, "No tour decoded"
    assert len(tour) == n, f"Tour length {len(tour)} != {n}"
    assert set(tour) == set(range(n)), f"Tour doesn't visit all cities: {tour}"
    assert len(set(tour)) == n, "Tour has repeated cities (subtour)"


def test_instance_original_bb():
    """B&B with all SECs: optimal = 90 min, tour visits all 5 cities."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    x_nn = tour_to_x_vector(nn_tour, N)

    model = build_model(DEFAULT_DIST, N, include_secs=True)
    result = BranchAndBound(model, strategy="best_first",
                            branching="most_infeasible",
                            initial_incumbent=nn_cost, initial_x=x_nn).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 90.0) < 1e-4, f"Expected 90, got {result.obj_value}"

    tour = decode_tour(result, N)
    _verify_tour(tour, N, DEFAULT_DIST)
    assert abs(tour_cost(tour, DEFAULT_DIST) - 90.0) < 1e-4

    print(f"PASS: bb_original | cost=90 | nodes={result.nodes_explored}")


def test_instance_original_bc():
    """B&C (lazy SECs): finds same optimal 90 min without pre-loading all SECs."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    x_nn = tour_to_x_vector(nn_tour, N)

    model = build_model(DEFAULT_DIST, N, include_secs=False)   # degree only
    bc = TSPBranchAndCut(N, model, strategy="best_first",
                         branching="most_infeasible",
                         cut_types=["subtour", "gomory"],
                         initial_incumbent=nn_cost, initial_x=x_nn)
    result = bc.solve()

    assert result.status == "optimal", f"B&C returned {result.status}"
    assert abs(result.obj_value - 90.0) < 1e-4, f"Expected 90, got {result.obj_value}"

    tour = decode_tour(result, N)
    _verify_tour(tour, N, DEFAULT_DIST)

    print(f"PASS: bc_lazy | cost=90 | nodes={result.nodes_explored}")


def test_bb_vs_bc_agree():
    """B&B (full model) and B&C (lazy) must find the same optimal cost."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    x_nn = tour_to_x_vector(nn_tour, N)

    model_full = build_model(DEFAULT_DIST, N, include_secs=True)
    model_lazy = build_model(DEFAULT_DIST, N, include_secs=False)

    r_bb = BranchAndBound(model_full, strategy="best_first",
                          initial_incumbent=nn_cost, initial_x=x_nn).solve()
    r_bc = TSPBranchAndCut(N, model_lazy, strategy="best_first",
                           cut_types=["subtour", "gomory"],
                           initial_incumbent=nn_cost, initial_x=x_nn).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4

    print(f"PASS: bb_vs_bc | cost={r_bb.obj_value:.0f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_nearest_neighbor_upper_bound():
    """Nearest-neighbor tour must be feasible and >= optimal."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    _verify_tour(nn_tour, N, DEFAULT_DIST)

    assert abs(tour_cost(nn_tour, DEFAULT_DIST) - nn_cost) < 1e-4

    model = build_model(DEFAULT_DIST, N, include_secs=True)
    result = BranchAndBound(model, strategy="best_first").solve()
    assert nn_cost >= result.obj_value - 1e-4, (
        f"NN heuristic {nn_cost} is less than optimal {result.obj_value}"
    )
    print(f"PASS: nn_upper_bound | NN={nn_cost:.0f} >= optimal={result.obj_value:.0f}")


def test_sensitivity_transition():
    """
    Vary d[C0][C2] = d[C2][C0]:
    - At d=10: must use C0-C2 arc, cost = 80
    - At d=30: must avoid C0-C2 arc, cost = 95 (constant)
    """
    sens = sensitivity_analysis([10.0, 30.0])
    (d10, cost10, tour10), (d30, cost30, tour30) = sens

    assert cost10 is not None and cost30 is not None
    _verify_tour(tour10, N, DEFAULT_DIST)
    _verify_tour(tour30, N, DEFAULT_DIST)

    # At d=10: should be 80 (= 10 + 70)
    assert abs(cost10 - 80.0) < 1e-4, f"Expected 80 at d=10, got {cost10}"

    # At d=30: should be 95 (avoids C0-C2)
    assert abs(cost30 - 95.0) < 1e-4, f"Expected 95 at d=30, got {cost30}"

    # At d=10: must use the C0-C2 arc (either direction)
    uses_02_at_10 = any(
        (tour10[k] == 0 and tour10[(k+1)%N] == 2) or
        (tour10[k] == 2 and tour10[(k+1)%N] == 0)
        for k in range(N)
    )
    assert uses_02_at_10, f"d=10: expected C0-C2 arc in tour, got {tour10}"

    print(f"PASS: sensitivity | d=10 cost={cost10:.0f} (uses C0-C2) | "
          f"d=30 cost={cost30:.0f} (avoids C0-C2)")


if __name__ == "__main__":
    test_instance_original_bb()
    test_instance_original_bc()
    test_bb_vs_bc_agree()
    test_nearest_neighbor_upper_bound()
    test_sensitivity_transition()
    print("\nAll P7 tests passed!")
