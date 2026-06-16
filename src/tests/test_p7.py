"""Tests for Problem 7 - TSP (PIB - MTZ/DFJ formulation)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p7_tsp import (
    build_model, build_model_mtz, decode_tour, nearest_neighbor_tour, tour_to_x_vector,
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
    """B&B with MTZ model: optimal = 70 min, tour visits all 6 cities."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    x_nn = tour_to_x_vector(nn_tour, N)

    model = build_model_mtz(DEFAULT_DIST, N)
    # Warm-start: embed tour x_vector into the longer MTZ variable vector
    x_mtz = np.zeros(N * N + (N - 1))
    x_mtz[:N * N] = x_nn

    result = BranchAndBound(model, strategy="best_first",
                            branching="most_infeasible",
                            initial_incumbent=nn_cost, initial_x=x_mtz).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 70.0) < 1e-4, f"Expected 70, got {result.obj_value}"

    tour = decode_tour(result, N)
    _verify_tour(tour, N, DEFAULT_DIST)
    assert abs(tour_cost(tour, DEFAULT_DIST) - 70.0) < 1e-4

    print(f"PASS: bb_mtz_original | cost=70 | nodes={result.nodes_explored}")


def test_instance_original_bc():
    """B&C (lazy DFJ SECs): finds same optimal 70 min without MTZ."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    x_nn = tour_to_x_vector(nn_tour, N)

    model = build_model(DEFAULT_DIST, N, include_secs=False)   # degree only
    bc = TSPBranchAndCut(N, model, strategy="best_first",
                         branching="most_infeasible",
                         cut_types=["subtour", "gomory"],
                         initial_incumbent=nn_cost, initial_x=x_nn)
    result = bc.solve()

    assert result.status == "optimal", f"B&C returned {result.status}"
    assert abs(result.obj_value - 70.0) < 1e-4, f"Expected 70, got {result.obj_value}"

    tour = decode_tour(result, N)
    _verify_tour(tour, N, DEFAULT_DIST)

    print(f"PASS: bc_lazy_original | cost=70 | nodes={result.nodes_explored}")


def test_bb_vs_bc_agree():
    """B&B (MTZ) and B&C (lazy DFJ) must find the same optimal cost."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    x_nn = tour_to_x_vector(nn_tour, N)

    model_mtz = build_model_mtz(DEFAULT_DIST, N)
    x_mtz = np.zeros(N * N + (N - 1))
    x_mtz[:N * N] = x_nn

    model_lazy = build_model(DEFAULT_DIST, N, include_secs=False)

    r_bb = BranchAndBound(model_mtz, strategy="best_first",
                          initial_incumbent=nn_cost, initial_x=x_mtz).solve()
    r_bc = TSPBranchAndCut(N, model_lazy, strategy="best_first",
                           cut_types=["subtour", "gomory"],
                           initial_incumbent=nn_cost, initial_x=x_nn).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4

    print(f"PASS: bb_vs_bc | cost={r_bb.obj_value:.0f} | "
          f"B&B(MTZ) nodes={r_bb.nodes_explored} | B&C(DFJ) nodes={r_bc.nodes_explored}")


def test_nearest_neighbor_upper_bound():
    """Nearest-neighbor tour must be feasible and >= optimal (NN=75, optimal=70)."""
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    _verify_tour(nn_tour, N, DEFAULT_DIST)
    assert abs(tour_cost(nn_tour, DEFAULT_DIST) - nn_cost) < 1e-4
    # NN from E0 = E0->E1->E2->E3->E4->E5->E0 = 10+12+8+11+9+25 = 75
    assert abs(nn_cost - 75.0) < 1e-4, f"Expected NN cost 75, got {nn_cost}"

    model = build_model(DEFAULT_DIST, N, include_secs=True)
    result = BranchAndBound(model, strategy="best_first").solve()
    assert nn_cost >= result.obj_value - 1e-4, (
        f"NN heuristic {nn_cost} is less than optimal {result.obj_value}"
    )
    print(f"PASS: nn_upper_bound | NN={nn_cost:.0f} >= optimal={result.obj_value:.0f}")


def test_sensitivity_transition():
    """
    Vary d[E0][E2] = d[E2][E0]:
    - At d=10: must use E0-E2 arc, cost = 10 + 55 = 65
    - At d=25: must avoid E0-E2 arc, cost = 71 (constant)
    """
    sens = sensitivity_analysis([10.0, 25.0])
    (d10, cost10, tour10), (d25, cost25, tour25) = sens

    assert cost10 is not None and cost25 is not None
    _verify_tour(tour10, N, DEFAULT_DIST)
    _verify_tour(tour25, N, DEFAULT_DIST)

    # At d=10: cost = 10 + 55 = 65
    assert abs(cost10 - 65.0) < 1e-4, f"Expected 65 at d=10, got {cost10}"

    # At d=25: avoid E0-E2, cost = 71
    assert abs(cost25 - 71.0) < 1e-4, f"Expected 71 at d=25, got {cost25}"

    # At d=10: must use the E0-E2 arc (either direction)
    uses_02_at_10 = any(
        (tour10[k] == 0 and tour10[(k+1)%N] == 2) or
        (tour10[k] == 2 and tour10[(k+1)%N] == 0)
        for k in range(N)
    )
    assert uses_02_at_10, f"d=10: expected E0-E2 arc in tour, got {tour10}"

    print(f"PASS: sensitivity | d=10 cost={cost10:.0f} (uses E0-E2) | "
          f"d=25 cost={cost25:.0f} (avoids E0-E2)")


if __name__ == "__main__":
    test_instance_original_bb()
    test_instance_original_bc()
    test_bb_vs_bc_agree()
    test_nearest_neighbor_upper_bound()
    test_sensitivity_transition()
    print("\nAll P7 tests passed!")
