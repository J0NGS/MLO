"""Tests for Problem 3 - Datacenter Flow Selection (Multidimensional Knapsack, PIB)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from problems.p3_datacenter_flows import build_model, random_instance, fractional_knapsack_bound
from core import BranchAndBound, BranchAndCut


def _verify_solution(result, banda, buffer, priority, banda_cap, buffer_cap):
    assert result.status == "optimal"
    assert result.x is not None
    n = len(banda)
    x = np.round(result.x).astype(int)

    # All variables binary
    assert all(v in (0, 1) for v in x), "Non-binary solution"

    total_banda  = sum(banda[j]   * x[j] for j in range(n))
    total_buffer = sum(buffer[j]  * x[j] for j in range(n))
    obj          = sum(priority[j] * x[j] for j in range(n))

    assert total_banda  <= banda_cap  + 1e-6, f"Banda violated: {total_banda} > {banda_cap}"
    assert total_buffer <= buffer_cap + 1e-6, f"Buffer violated: {total_buffer} > {buffer_cap}"
    assert abs(obj - result.obj_value) < 1e-4, f"Obj mismatch: computed={obj} reported={result.obj_value}"


def test_instance_original():
    """Assignment instance: 7 flows, banda=100, buffer=25. Expected priority=165 (F5+F6+F7)."""
    banda    = [30, 20, 45, 15, 35, 25, 40]
    buffer   = [8,  5,  10,  3,  9,  6,  7]
    priority = [50, 30, 70, 20, 60, 40, 65]
    banda_cap, buffer_cap = 100, 25

    model = build_model(banda, buffer, priority, banda_cap, buffer_cap)
    solver = BranchAndBound(model, strategy="best_first", branching="most_infeasible")
    result = solver.solve()

    _verify_solution(result, banda, buffer, priority, banda_cap, buffer_cap)
    assert abs(result.obj_value - 165.0) < 1e-4, f"Expected 165, got {result.obj_value}"

    # Check that F5, F6, F7 (indices 4, 5, 6) are selected
    selected = set(j for j in range(7) if round(result.x[j]) == 1)
    assert selected == {4, 5, 6}, f"Expected {{F5,F6,F7}}, got {{'F'+str(j+1) for j in selected}}"
    print(f"PASS: instance_original | obj={result.obj_value:.1f} | nodes={result.nodes_explored}")


def test_fractional_bound_is_upper_bound():
    """Fractional knapsack bound must be >= optimal integer objective."""
    banda    = [30, 20, 45, 15, 35, 25, 40]
    buffer   = [8,  5,  10,  3,  9,  6,  7]
    priority = [50, 30, 70, 20, 60, 40, 65]
    banda_cap, buffer_cap = 100, 25

    fk = fractional_knapsack_bound(banda, buffer, priority, banda_cap, buffer_cap)
    model = build_model(banda, buffer, priority, banda_cap, buffer_cap)
    result = BranchAndBound(model, strategy="best_first").solve()

    assert fk >= result.obj_value - 1e-6, \
        f"Fractional bound {fk} < integer optimal {result.obj_value}"
    print(f"PASS: fractional_bound | fk={fk:.2f} >= opt={result.obj_value:.1f}")


def test_bb_vs_bc_same_result():
    """B&B and B&C (cover cuts) must find the same optimal value."""
    banda    = [30, 20, 45, 15, 35, 25, 40]
    buffer   = [8,  5,  10,  3,  9,  6,  7]
    priority = [50, 30, 70, 20, 60, 40, 65]
    banda_cap, buffer_cap = 100, 25

    model = build_model(banda, buffer, priority, banda_cap, buffer_cap)
    r_bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible").solve()
    r_bc = BranchAndCut(model,  strategy="best_first", branching="most_infeasible",
                        cut_types=["cover"]).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, \
        f"B&B={r_bb.obj_value} vs B&C={r_bc.obj_value}"
    print(f"PASS: bb_vs_bc | B&B={r_bb.obj_value:.1f} nodes={r_bb.nodes_explored} "
          f"| B&C={r_bc.obj_value:.1f} nodes={r_bc.nodes_explored}")


def test_instance_random_medium():
    """Random 20-flow instance: verify feasibility and that B&B == B&C."""
    banda, buffer, priority = random_instance(20, seed=2)
    n = 20
    # Scale capacities proportionally
    banda_cap  = int(100 * n/7 * 0.6)
    buffer_cap = int(25  * n/7 * 0.6)

    model = build_model(banda, buffer, priority, banda_cap, buffer_cap)
    r_bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible").solve()
    r_bc = BranchAndCut(model,  strategy="best_first", branching="most_infeasible",
                        cut_types=["cover"]).solve()

    _verify_solution(r_bb, banda, buffer, priority, banda_cap, buffer_cap)
    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, \
        f"B&B={r_bb.obj_value} vs B&C={r_bc.obj_value}"
    print(f"PASS: random_medium | n=20 | opt={r_bb.obj_value:.1f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_tight_capacity():
    """When capacity is very tight, only the smallest flows are selected."""
    banda    = [10, 20, 30, 40, 50]
    buffer   = [2,  4,  6,  8,  10]
    priority = [5,  10, 15, 20, 25]  # priority proportional to size
    banda_cap, buffer_cap = 15, 3   # only the smallest flow (F1) fits

    model = build_model(banda, buffer, priority, banda_cap, buffer_cap)
    result = BranchAndBound(model, strategy="best_first").solve()

    _verify_solution(result, banda, buffer, priority, banda_cap, buffer_cap)
    assert abs(result.obj_value - 5.0) < 1e-4, f"Expected 5.0, got {result.obj_value}"
    selected = [j for j in range(5) if round(result.x[j]) == 1]
    assert selected == [0], f"Expected only F1, got {selected}"
    print(f"PASS: tight_capacity | obj={result.obj_value:.1f} | selected=F1 only")


if __name__ == "__main__":
    test_instance_original()
    test_fractional_bound_is_upper_bound()
    test_bb_vs_bc_same_result()
    test_instance_random_medium()
    test_tight_capacity()
    print("\nAll tests passed!")
