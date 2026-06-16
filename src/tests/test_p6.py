"""Tests for Problem 6 - Network Zone Coverage (PIB Set Covering)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p6_network_coverage import (
    build_model, decode_solution, greedy_set_cover, greedy_to_x_vector,
    random_instance, sensitivity_analysis,
    DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES, N_SITES, SITE_NAMES,
)
from core import BranchAndBound, BranchAndCut


def _verify_coverage(selected, coverage, n_zones):
    """Every zone must be covered by at least one selected site."""
    covered = set()
    for j in selected:
        covered |= coverage[j]
    assert covered == set(range(n_zones)), (
        f"Not all zones covered: missing {set(range(n_zones)) - covered}"
    )


def test_instance_original():
    """Optimal must be cost=10 with {A1, A2, A6} (indices 0, 1, 5)."""
    model = build_model(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES, SITE_NAMES)

    g_sel, g_cost = greedy_set_cover(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES)
    x_g = greedy_to_x_vector(g_sel, N_SITES)

    result = BranchAndBound(model, strategy="best_first",
                            branching="most_infeasible",
                            initial_incumbent=g_cost, initial_x=x_g).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 10.0) < 1e-4, f"Expected cost 10, got {result.obj_value}"

    selected = decode_solution(result, N_SITES)
    _verify_coverage(selected, DEFAULT_COVERAGE, N_ZONES)

    # Must include exactly 3 antenas; A1(0), A2(1), A6(5) is the unique min-cost solution
    assert len(selected) == 3
    assert set(selected) == {0, 1, 5}, (
        f"Expected {{A1,A2,A6}} = {{0,1,5}}, got {{{','.join(SITE_NAMES[j] for j in selected)}}}"
    )
    print(f"PASS: instance_original | cost=10 | nodes={result.nodes_explored}")


def test_bb_vs_bc_same_result():
    """B&B and B&C must find the same optimal cost."""
    model = build_model(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES, SITE_NAMES)

    g_sel, g_cost = greedy_set_cover(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES)
    x_g = greedy_to_x_vector(g_sel, N_SITES)

    r_bb = BranchAndBound(model, strategy="best_first",
                          initial_incumbent=g_cost, initial_x=x_g).solve()
    r_bc = BranchAndCut(model, strategy="best_first", cut_types=["gomory"],
                        initial_incumbent=g_cost, initial_x=x_g).solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, (
        f"B&B={r_bb.obj_value} != B&C={r_bc.obj_value}"
    )
    # Both solutions must be feasible
    _verify_coverage(decode_solution(r_bb, N_SITES), DEFAULT_COVERAGE, N_ZONES)
    _verify_coverage(decode_solution(r_bc, N_SITES), DEFAULT_COVERAGE, N_ZONES)
    print(f"PASS: bb_vs_bc | cost={r_bb.obj_value:.1f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored}")


def test_greedy_is_upper_bound():
    """Greedy cost must be >= optimal and the greedy solution must be feasible."""
    g_sel, g_cost = greedy_set_cover(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES)
    _verify_coverage(g_sel, DEFAULT_COVERAGE, N_ZONES)

    model = build_model(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES)
    result = BranchAndBound(model, strategy="best_first").solve()

    assert result.obj_value <= g_cost + 1e-6, (
        f"Greedy {g_cost} must be >= optimal {result.obj_value}"
    )
    print(f"PASS: greedy_upper_bound | greedy={g_cost:.1f} >= optimal={result.obj_value:.1f}")


def test_sensitivity_switch():
    """
    Sensitivity on cost of A1 (site 0):
    - c(A1) <= 5: {A1,A2,A6} is optimal  -> A1 selected
    - c(A1) >= 7: {A3,A4,A6} is optimal  -> A1 NOT selected, cost=13 constant
    """
    sens = sensitivity_analysis(
        cost_range=[1.0, 3.0, 5.0, 7.0, 9.0],
        site_idx=0,
    )

    for c_val, total, selected in sens:
        assert total is not None, f"No solution at c(A1)={c_val}"
        _verify_coverage(selected, DEFAULT_COVERAGE, N_ZONES)

        if c_val <= 5.0:
            # A1 must be in optimal (strictly cheaper to use it)
            assert 0 in selected, (
                f"c(A1)={c_val}: expected A1 selected, got {selected}"
            )
            assert abs(total - (c_val + 7.0)) < 1e-4, (
                f"c(A1)={c_val}: expected cost {c_val+7}, got {total}"
            )
        elif c_val >= 7.0:
            # A1 is too expensive; {A3,A4,A6} dominates
            assert 0 not in selected, (
                f"c(A1)={c_val}: A1 should NOT be selected, got {selected}"
            )
            assert abs(total - 13.0) < 1e-4, (
                f"c(A1)={c_val}: expected fixed cost 13, got {total}"
            )

    print("PASS: sensitivity_switch | A1<=5: c1+7, A1>=7: 13 (A3+A4+A6)")


def test_random_instance_feasible():
    """Random instances: B&B solution must cover all zones at a cost <= greedy."""
    for n_z, n_s, seed in [(10, 8, 1), (15, 10, 2)]:
        costs, coverage = random_instance(n_z, n_s, zones_per_site=4, seed=seed)
        sn = [f"S{j+1}" for j in range(n_s)]

        g_sel, g_cost = greedy_set_cover(costs, coverage, n_z)
        x_g = greedy_to_x_vector(g_sel, n_s)

        model = build_model(costs, coverage, n_z, sn)
        result = BranchAndBound(model, strategy="best_first",
                                branching="most_infeasible",
                                initial_incumbent=g_cost, initial_x=x_g,
                                time_limit=30.0).solve()

        assert result.status == "optimal", f"n_z={n_z}: got {result.status}"
        assert result.obj_value <= g_cost + 1e-6

        selected = decode_solution(result, n_s)
        _verify_coverage(selected, coverage, n_z)

        print(f"PASS: random n_z={n_z} n_s={n_s} | "
              f"cost={result.obj_value:.1f} | nodes={result.nodes_explored}")


if __name__ == "__main__":
    test_instance_original()
    test_bb_vs_bc_same_result()
    test_greedy_is_upper_bound()
    test_sensitivity_switch()
    test_random_instance_feasible()
    print("\nAll P6 tests passed!")
