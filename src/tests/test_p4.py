"""Tests for Problem 4 - VM Scheduling / Bin Packing (PLIP)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from problems.p4_vm_binpacking import (
    build_model, decode_solution, ffd_to_x_vector,
    first_fit_decreasing, lb_fractional, random_instance,
)
from core import BranchAndBound, BranchAndCut


def _warm_start(demands, capacity):
    """Return (n_ffd, x_ffd) for warm-starting solvers."""
    n_ffd, bins_ffd = first_fit_decreasing(demands, capacity)
    x_ffd = ffd_to_x_vector(bins_ffd, len(demands), n_ffd)
    return n_ffd, x_ffd, bins_ffd


def _verify_solution(result, demands, capacity, max_servers):
    assert result.status == "optimal", f"Expected optimal, got {result.status}"
    assert result.x is not None
    n = len(demands)
    k = max_servers

    x = np.round(result.x).astype(int)
    # All variables are binary
    assert all(v in (0, 1) for v in x)

    # Each VM assigned to exactly one server
    for i in range(n):
        total = sum(x[k + i * k + j] for j in range(k))
        assert total == 1, f"VM{i+1} assigned to {total} servers"

    # Capacity constraints: sum_i d_i * x_ij <= capacity * y_j
    for j in range(k):
        load = sum(demands[i] * x[k + i * k + j] for i in range(n))
        cap_j = capacity * x[j]
        assert load <= cap_j + 1e-6, f"Server {j+1} overloaded: {load} > {cap_j}"


def test_instance_original():
    """Assignment instance: 6 VMs, capacity=10. LB=3, FFD=3, optimal=3."""
    demands = [4.0, 3.0, 5.0, 2.0, 4.0, 3.0]
    cap = 10.0
    n_ffd, x_ffd, _ = _warm_start(demands, cap)

    assert lb_fractional(demands, cap) == 3
    assert n_ffd == 3

    model = build_model(demands, cap, max_servers=n_ffd)
    result = BranchAndBound(model, strategy="best_first", branching="most_infeasible",
                            initial_incumbent=float(n_ffd), initial_x=x_ffd).solve()

    _verify_solution(result, demands, cap, n_ffd)
    assert round(result.obj_value) == 3, f"Expected 3 servers, got {result.obj_value}"
    print(f"PASS: instance_original | bins=3 | nodes={result.nodes_explored}")


def test_ffd_optimality():
    """When LB == FFD, warm-start proves optimality instantly (few nodes)."""
    demands = [2.0, 5.0, 1.0, 3.0, 1.0, 4.0, 4.0, 4.0, 4.0, 2.0]  # n=10
    cap = 10.0
    n_ffd, x_ffd, _ = _warm_start(demands, cap)
    lb = lb_fractional(demands, cap)

    assert lb == n_ffd, f"LB={lb} != FFD={n_ffd}, test assumption fails"

    model = build_model(demands, cap, max_servers=n_ffd)
    result = BranchAndBound(model, strategy="best_first",
                            initial_incumbent=float(n_ffd), initial_x=x_ffd).solve()

    assert result.status == "optimal"
    assert round(result.obj_value) == n_ffd
    # Warm start + tight LP = root node prunes immediately (very few nodes)
    assert result.nodes_explored <= 5, f"Expected <= 5 nodes, got {result.nodes_explored}"
    print(f"PASS: ffd_optimality | n=10 | bins={n_ffd} | nodes={result.nodes_explored}")


def test_bb_vs_bc_same_result():
    """B&B and B&C must find the same number of bins."""
    demands = [4.0, 3.0, 5.0, 2.0, 4.0, 3.0]
    cap = 10.0
    n_ffd, x_ffd, _ = _warm_start(demands, cap)

    model = build_model(demands, cap, max_servers=n_ffd)
    r_bb = BranchAndBound(model, strategy="best_first",
                          initial_incumbent=float(n_ffd), initial_x=x_ffd).solve()
    r_bc = BranchAndCut(model, strategy="best_first", cut_types=["cover"],
                        initial_incumbent=float(n_ffd), initial_x=x_ffd).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert round(r_bb.obj_value) == round(r_bc.obj_value), \
        f"B&B={r_bb.obj_value} vs B&C={r_bc.obj_value}"
    print(f"PASS: bb_vs_bc | bins={round(r_bb.obj_value)} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_ffd_feasibility():
    """FFD solution must satisfy all capacity and assignment constraints."""
    demands = [4.0, 3.0, 5.0, 2.0, 4.0, 3.0]
    cap = 10.0
    n_ffd, bins_ffd = first_fit_decreasing(demands, cap)

    # Check assignment (each VM appears exactly once)
    all_vms = sorted(vm for b in bins_ffd for vm in b)
    assert all_vms == list(range(len(demands))), "Some VM missing or duplicated in FFD"

    # Check capacity
    for j, vms in enumerate(bins_ffd):
        load = sum(demands[i] for i in vms)
        assert load <= cap + 1e-6, f"FFD bin {j+1} overloaded: {load}"

    print(f"PASS: ffd_feasibility | bins={n_ffd} | loads={[sum(demands[i] for i in b) for b in bins_ffd]}")


def test_single_vm_per_bin():
    """When all VMs exceed half capacity, each goes to its own bin."""
    demands = [6.0, 7.0, 8.0, 9.0]  # no two fit together in cap=10
    cap = 10.0
    n_ffd, x_ffd, _ = _warm_start(demands, cap)

    assert n_ffd == 4, f"Expected 4 bins, FFD gave {n_ffd}"

    model = build_model(demands, cap, max_servers=n_ffd)
    result = BranchAndBound(model, strategy="best_first",
                            initial_incumbent=float(n_ffd), initial_x=x_ffd).solve()

    assert result.status == "optimal"
    assert round(result.obj_value) == 4
    _verify_solution(result, demands, cap, n_ffd)
    print(f"PASS: single_vm_per_bin | bins=4 | nodes={result.nodes_explored}")


if __name__ == "__main__":
    test_instance_original()
    test_ffd_optimality()
    test_bb_vs_bc_same_result()
    test_ffd_feasibility()
    test_single_vm_per_bin()
    print("\nAll tests passed!")
