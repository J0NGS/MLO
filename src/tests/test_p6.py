"""Tests for Problem 6 - Network Zone Coverage (PIB Set Covering)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from problems.p6_network_coverage import (
    build_model, decode_solution, greedy_set_cover, greedy_to_x_vector,
    random_instance, sensitivity_analysis, CoverageBC,
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
    """Optimal must be cost=230 with {L2, L3} (indices 1, 2)."""
    model = build_model(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES, SITE_NAMES)

    g_sel, g_cost = greedy_set_cover(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES)
    x_g = greedy_to_x_vector(g_sel, N_SITES)

    result = BranchAndBound(model, strategy="best_first",
                            branching="most_infeasible",
                            initial_incumbent=g_cost, initial_x=x_g).solve()

    assert result.status == "optimal"
    assert abs(result.obj_value - 230.0) < 1e-4, f"Expected cost 230, got {result.obj_value}"

    selected = decode_solution(result, N_SITES)
    _verify_coverage(selected, DEFAULT_COVERAGE, N_ZONES)

    # Must include {L2, L3} (indices 1, 2)
    assert set(selected) == {1, 2}, (
        f"Expected {{L2,L3}} = {{1,2}}, got {{{','.join(SITE_NAMES[j] for j in selected)}}}"
    )
    print(f"PASS: instance_original | cost=230 | nodes={result.nodes_explored}")


def test_bb_vs_bc_same_result():
    """B&B and B&C must find the same optimal cost."""
    model = build_model(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES, SITE_NAMES)

    g_sel, g_cost = greedy_set_cover(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES)
    x_g = greedy_to_x_vector(g_sel, N_SITES)

    r_bb = BranchAndBound(model, strategy="best_first",
                          initial_incumbent=g_cost, initial_x=x_g).solve()
    bc_solver = CoverageBC(N_ZONES, N_SITES, DEFAULT_COVERAGE,
                           model, strategy="best_first", cut_types=["gomory", "clique"],
                           initial_incumbent=g_cost, initial_x=x_g)
    r_bc = bc_solver.solve()

    assert r_bb.status == r_bc.status == "optimal"
    assert abs(r_bb.obj_value - r_bc.obj_value) < 1e-4, (
        f"B&B={r_bb.obj_value} != B&C={r_bc.obj_value}"
    )
    _verify_coverage(decode_solution(r_bb, N_SITES), DEFAULT_COVERAGE, N_ZONES)
    _verify_coverage(decode_solution(r_bc, N_SITES), DEFAULT_COVERAGE, N_ZONES)

    clique_cuts = sum(1 for e in bc_solver._cut_log if e["type"] == "clique")
    print(f"PASS: bb_vs_bc | cost={r_bb.obj_value:.1f} | "
          f"B&B nodes={r_bb.nodes_explored} | B&C nodes={r_bc.nodes_explored} | "
          f"clique cuts added={clique_cuts} (expected 0)")


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
    Sensitivity on cost of L3 (site 2):
    - c(L3) = 150: {L2,L3} is optimal, cost = 230
    - c(L3) = 300: {L1,L4,L5} is optimal, cost = 320
    """
    sens = sensitivity_analysis(
        cost_range=[150.0, 300.0],
        site_idx=2,
    )

    (c150, total150, sel150), (c300, total300, sel300) = sens

    assert total150 is not None and total300 is not None
    _verify_coverage(sel150, DEFAULT_COVERAGE, N_ZONES)
    _verify_coverage(sel300, DEFAULT_COVERAGE, N_ZONES)

    # At L3=150: {L2,L3} optimal, cost=230
    assert set(sel150) == {1, 2}, f"c(L3)=150: expected {{L2,L3}}, got {sel150}"
    assert abs(total150 - 230.0) < 1e-4, f"c(L3)=150: expected 230, got {total150}"

    # At L3=300: {L1,L4,L5} optimal, cost=320
    assert set(sel300) == {0, 3, 4}, f"c(L3)=300: expected {{L1,L4,L5}}, got {sel300}"
    assert abs(total300 - 320.0) < 1e-4, f"c(L3)=300: expected 320, got {total300}"

    print("PASS: sensitivity_switch | L3=150: {L2,L3}=230, L3=300: {L1,L4,L5}=320")


def test_clique_cuts_never_violated():
    """Clique cuts must never be violated by any LP relaxation solution."""
    from problems.p6_network_coverage import zone_conflict_clique_cuts
    from core.relaxation import solve_relaxation

    model = build_model(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES, SITE_NAMES)
    lp = solve_relaxation(model)
    assert lp.status == "optimal"

    cuts = zone_conflict_clique_cuts(lp.x, DEFAULT_COVERAGE, N_SITES, N_ZONES)
    assert len(cuts) == 0, (
        f"Expected 0 violated clique cuts, but got {len(cuts)}: {cuts}"
    )
    print(f"PASS: clique_cuts_never_violated | 0 violated cuts (LP relaxation satisfies all)")


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
    test_clique_cuts_never_violated()
    test_random_instance_feasible()
    print("\nAll P6 tests passed!")
