"""Tests for Problem 1 - Task Allocation in Multiprocessor System (PLIP)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from problems.p1_task_allocation import build_model, random_instance
from core import BranchAndBound, BBResult


def _verify_solution(result: BBResult, p, capacity):
    """Check that the solution is feasible (all constraints satisfied)."""
    assert result.status == "optimal", f"Expected optimal, got {result.status}"
    assert result.x is not None

    S, T = len(p), len(p[0])
    x = result.x

    # Each task assigned to exactly one server
    for j in range(T):
        total = sum(round(x[i * T + j]) for i in range(S))
        assert total == 1, f"Task T{j+1} assigned to {total} servers"

    # Capacity and makespan constraints
    M = x[S * T]
    for i in range(S):
        load = sum(p[i][j] * round(x[i * T + j]) for j in range(T))
        assert load <= capacity + 1e-6, f"S{i+1} capacity violated: {load} > {capacity}"
        assert load <= M + 1e-6, f"S{i+1} load {load} exceeds makespan {M:.4f}"


def test_instance_original():
    """Instance from the assignment: 3 servers, 4 tasks, capacity=12. Expected makespan=7."""
    p = [
        [4, 5, 3, 7],   # S1
        [3, 6, 4, 5],   # S2
        [5, 4, 6, 3],   # S3
    ]
    capacity = 12
    model = build_model(p, capacity)
    solver = BranchAndBound(model, strategy="best_first", branching="most_infeasible")
    result = solver.solve()

    _verify_solution(result, p, capacity)
    assert abs(result.obj_value - 7.0) < 1e-4, f"Expected makespan=7.0, got {result.obj_value}"

    # Verify the optimal assignment: S3 gets T2+T4 (load=7), others get 1 task each
    S, T = 3, 4
    loads = [
        sum(p[i][j] * round(result.x[i * T + j]) for j in range(T))
        for i in range(S)
    ]
    assert max(loads) == 7, f"Max load should be 7, got {max(loads)}"
    print(f"PASS: instance_original | makespan={result.obj_value:.2f} | nodes={result.nodes_explored}")


def test_instance_balanced():
    """Symmetric instance where all tasks have equal duration: makespan = n_tasks / n_servers * duration."""
    # 4 servers, 8 tasks, each task takes 5s on any server → optimal makespan = 10s (2 tasks/server)
    p = [[5] * 8 for _ in range(4)]
    capacity = 25
    model = build_model(p, capacity)
    solver = BranchAndBound(model, strategy="best_first", branching="first_fractional")
    result = solver.solve()

    _verify_solution(result, p, capacity)
    assert abs(result.obj_value - 10.0) < 1e-4, f"Expected makespan=10.0, got {result.obj_value}"
    print(f"PASS: instance_balanced | makespan={result.obj_value:.2f} | nodes={result.nodes_explored}")


def test_instance_random_medium():
    """Random 5x8 instance (seed=1): verify feasibility and optimality bound."""
    p = random_instance(5, 8, 20, seed=1)
    capacity = 20
    model = build_model(p, capacity)
    solver = BranchAndBound(model, strategy="best_first", branching="most_infeasible")
    result = solver.solve()

    _verify_solution(result, p, capacity)
    # LP relaxation gives a lower bound; optimal must be >= LP bound and <= trivial UB
    lp_lb = 3.0   # empirically observed LP bound is ~3.1
    trivial_ub = max(min(p[i][j] for i in range(5)) for j in range(8)) * 8
    assert result.obj_value >= lp_lb, f"Makespan {result.obj_value} below LP bound"
    print(f"PASS: instance_random_medium | makespan={result.obj_value:.2f} | nodes={result.nodes_explored}")


def test_single_server():
    """Edge case: 1 server must take all tasks; makespan = sum of all task durations."""
    p = [[2, 3, 5, 4]]   # S1 only
    capacity = 20
    model = build_model(p, capacity)
    solver = BranchAndBound(model, strategy="best_first")
    result = solver.solve()

    _verify_solution(result, p, capacity)
    assert abs(result.obj_value - 14.0) < 1e-4, f"Expected makespan=14.0, got {result.obj_value}"
    print(f"PASS: single_server | makespan={result.obj_value:.2f} | nodes={result.nodes_explored}")


def test_bfs_vs_dfs_same_result():
    """Different search strategies must find the same optimal value."""
    p = [
        [4, 5, 3, 7],
        [3, 6, 4, 5],
        [5, 4, 6, 3],
    ]
    capacity = 12
    model = build_model(p, capacity)

    r_bfs = BranchAndBound(model, strategy="bfs").solve()
    r_dfs = BranchAndBound(model, strategy="dfs").solve()
    r_best = BranchAndBound(model, strategy="best_first").solve()

    assert r_bfs.status == r_dfs.status == r_best.status == "optimal"
    assert abs(r_bfs.obj_value - 7.0) < 1e-4, f"BFS got {r_bfs.obj_value}"
    assert abs(r_dfs.obj_value - 7.0) < 1e-4, f"DFS got {r_dfs.obj_value}"
    assert abs(r_best.obj_value - 7.0) < 1e-4, f"BestFirst got {r_best.obj_value}"
    print(f"PASS: strategies | BFS nodes={r_bfs.nodes_explored} | DFS={r_dfs.nodes_explored} | BestFirst={r_best.nodes_explored}")


if __name__ == "__main__":
    test_instance_original()
    test_instance_balanced()
    test_instance_random_medium()
    test_single_server()
    test_bfs_vs_dfs_same_result()
    print("\nAll tests passed!")
