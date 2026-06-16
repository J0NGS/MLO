"""Tests for Problem 2 - Project Selection (PIB)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from problems.p2_project_selection import build_model
from core import BranchAndBound, BranchAndCut


def test_instance_original():
    """Instance from the assignment: 6 projects, budget=280."""
    model = build_model()
    solver = BranchAndBound(model, strategy="best_first")
    result = solver.solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 405) < 1, f"Expected 405 pts, got {result.obj_value}"

    selected = [i for i, xi in enumerate(result.x) if round(xi) == 1]
    # Optimal: P1, P2, P3, P4 (0-indexed: 0,1,2,3)
    assert set(selected) == {0, 1, 2, 3}, f"Wrong projects selected: {selected}"

    # Verify constraints
    costs = [80, 60, 90, 50, 70, 100]
    total_cost = sum(costs[i] for i in selected)
    assert total_cost <= 280, f"Budget violated: {total_cost}"

    # Logical constraints
    x = np.round(result.x).astype(int)
    assert x[2] <= x[0], "P3 selected without P1"
    assert x[3] + x[4] <= 1, "P4 and P5 both selected"
    assert x[0] + x[1] + x[3] >= 2, "Fewer than 2 of {P1,P2,P4} selected"
    print(f"PASS: instance_original | obj={result.obj_value:.0f} | nodes={result.nodes_explored}")


def test_bb_vs_bc_same_result():
    """B&B and B&C must find the same optimal."""
    model = build_model()
    bb = BranchAndBound(model, strategy="best_first")
    bc = BranchAndCut(model, strategy="best_first", cut_types=["gomory"])

    r_bb = bb.solve()
    r_bc = bc.solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, \
        f"B&B={r_bb.obj_value} vs B&C={r_bc.obj_value}"
    print(f"PASS: bb_vs_bc | B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_instance_random_small():
    """Random instance: 8 projects, budget 400."""
    import random
    rng = random.Random(42)
    n = 8
    costs  = [rng.randint(40, 120) for _ in range(n)]
    impact = [rng.randint(50, 200) for _ in range(n)]
    budget = 400

    from core import MIPModel
    A_ub = [costs]
    b_ub = [budget]
    model = MIPModel(
        c=impact,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, 1)] * n,
        integrality=[2] * n,
        sense="max",
        var_names=[f"P{i+1}" for i in range(n)],
    )
    solver = BranchAndBound(model, strategy="best_first")
    result = solver.solve()

    assert result.status == "optimal"
    total_cost = sum(costs[i] * round(result.x[i]) for i in range(n))
    assert total_cost <= budget
    print(f"PASS: random_small | obj={result.obj_value:.0f} | nodes={result.nodes_explored}")


if __name__ == "__main__":
    test_instance_original()
    test_bb_vs_bc_same_result()
    test_instance_random_small()
    print("\nAll tests passed!")
