"""
Problema 6 - Cobertura de Zonas de Rede [PIB - Set Covering]

Uma empresa de telecomunicacoes precisa instalar antenas para cobrir
todas as 7 zonas de uma cidade. Ha 6 locais candidatos, cada um com
custo fixo e um subconjunto de zonas cobertas.

Formulacao PIB (Set Covering):
    min  sum_j c_j * x_j
    s.t.
    (1)  sum_{j: i in S_j} x_j >= 1    para cada zona i = 1..7
         x_j in {0,1}

Reescrita para linprog (A_ub * x <= b_ub):
    -sum_{j: i in S_j} x_j <= -1       para cada zona i

Instancia original:
    Antena  Custo   Zonas cobertas
    A1        3     Z1, Z2, Z5
    A2        5     Z2, Z3, Z4
    A3        4     Z1, Z4, Z6
    A4        7     Z3, Z5, Z7
    A5        6     Z5, Z6, Z7
    A6        2     Z2, Z6, Z7

Solucao otima: {A1, A2, A6}, custo total = 10.

Analise de sensibilidade (custo de A1 em [1..10]):
    c1 <= 5 -> otimo = {A1, A2, A6}, custo = c1 + 7
    c1  = 6 -> empate: {A1,A2,A6} = {A3,A4,A6} = 13
    c1 >= 7 -> otimo = {A3, A4, A6}, custo = 13
"""
from __future__ import annotations
import random
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult


# ------------------------------------------------------------------
# Default instance data
# ------------------------------------------------------------------

N_ZONES = 7
N_SITES = 6

ZONE_NAMES = [f"Z{i+1}" for i in range(N_ZONES)]
SITE_NAMES = [f"A{j+1}" for j in range(N_SITES)]

DEFAULT_COSTS: list[float] = [3.0, 5.0, 4.0, 7.0, 6.0, 2.0]

# coverage[j] = frozenset of zone indices (0-based) covered by site j
DEFAULT_COVERAGE: list[frozenset] = [
    frozenset({0, 1, 4}),   # A1: Z1, Z2, Z5
    frozenset({1, 2, 3}),   # A2: Z2, Z3, Z4
    frozenset({0, 3, 5}),   # A3: Z1, Z4, Z6
    frozenset({2, 4, 6}),   # A4: Z3, Z5, Z7
    frozenset({4, 5, 6}),   # A5: Z5, Z6, Z7
    frozenset({1, 5, 6}),   # A6: Z2, Z6, Z7
]


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    costs: list[float],
    coverage: list[frozenset],
    n_zones: int,
    site_names: list[str] | None = None,
) -> MIPModel:
    """
    Build Set Covering MIPModel.

    Coverage constraints are stored as -sum x_j <= -1 (equiv. sum x_j >= 1).
    """
    n_sites = len(costs)
    if site_names is None:
        site_names = [f"s{j+1}" for j in range(n_sites)]

    bounds = [(0.0, 1.0)] * n_sites
    integrality = [2] * n_sites  # all binary (PIB)

    A_ub, b_ub = [], []
    for i in range(n_zones):
        row = [(-1.0 if i in coverage[j] else 0.0) for j in range(n_sites)]
        A_ub.append(row)
        b_ub.append(-1.0)   # -sum x_j <= -1  <=>  sum x_j >= 1

    return MIPModel(
        c=list(costs),
        A_ub=A_ub, b_ub=b_ub,
        bounds=bounds, integrality=integrality,
        sense="min",
        var_names=site_names,
    )


# ------------------------------------------------------------------
# Greedy heuristic: min cost-per-newly-covered-zone
# ------------------------------------------------------------------

def greedy_set_cover(
    costs: list[float],
    coverage: list[frozenset],
    n_zones: int,
) -> tuple[list[int], float]:
    """
    Returns (selected_indices, total_cost).
    Greedy approximation: pick the site with lowest cost / new zones at each step.
    """
    uncovered = set(range(n_zones))
    selected: list[int] = []

    while uncovered:
        best_j, best_ratio = -1, float("inf")
        for j, cov in enumerate(coverage):
            if j in selected:
                continue
            new_zones = cov & uncovered
            if not new_zones:
                continue
            ratio = costs[j] / len(new_zones)
            if ratio < best_ratio:
                best_ratio, best_j = ratio, j
        if best_j < 0:
            break
        selected.append(best_j)
        uncovered -= coverage[best_j]

    return selected, sum(costs[j] for j in selected)


def greedy_to_x_vector(selected: list[int], n_sites: int) -> np.ndarray:
    x = np.zeros(n_sites)
    for j in selected:
        x[j] = 1.0
    return x


# ------------------------------------------------------------------
# Random instance generator
# ------------------------------------------------------------------

def random_instance(
    n_zones: int,
    n_sites: int,
    zones_per_site: int = 3,
    max_cost: float = 10.0,
    seed: int = 0,
) -> tuple[list[float], list[frozenset]]:
    """
    Random Set Covering instance. Guarantees every zone is covered
    by at least one site.
    """
    rng = random.Random(seed)
    costs = [round(rng.uniform(1.0, max_cost), 1) for _ in range(n_sites)]

    coverage: list[frozenset] = []
    for _ in range(n_sites):
        k = min(zones_per_site, n_zones)
        coverage.append(frozenset(rng.sample(range(n_zones), k)))

    # Ensure every zone is covered
    all_covered = set().union(*coverage)
    for i in set(range(n_zones)) - all_covered:
        j = rng.randint(0, n_sites - 1)
        coverage[j] = coverage[j] | {i}

    return costs, coverage


# ------------------------------------------------------------------
# Solution decoder and printer
# ------------------------------------------------------------------

def decode_solution(result: BBResult, n_sites: int) -> list[int]:
    if result.x is None:
        return []
    return [j for j in range(n_sites) if round(result.x[j]) == 1]


def print_solution(
    result: BBResult,
    costs: list[float],
    coverage: list[frozenset],
    n_zones: int,
    label: str = "",
    site_names: list[str] | None = None,
    zone_names: list[str] | None = None,
):
    n_sites = len(costs)
    if site_names is None:
        site_names = [f"S{j+1}" for j in range(n_sites)]
    if zone_names is None:
        zone_names = [f"Z{i+1}" for i in range(n_zones)]

    print("\n" + "=" * 65)
    if label:
        print(f"  {label}")
    print(f"Status          : {result.status}")
    print(f"Nos explorados  : {result.nodes_explored}")
    print(f"Tempo (s)       : {result.elapsed:.4f}")

    if result.x is None:
        print("Sem solucao viavel.")
        print("=" * 65)
        return

    selected = decode_solution(result, n_sites)
    covered = set().union(*(coverage[j] for j in selected)) if selected else set()
    print(f"Custo otimo     : {result.obj_value:.1f}")
    print(f"Antenas ativas  : {{{', '.join(site_names[j] for j in selected)}}}")
    print(f"Cobertura       : {'OK' if covered == set(range(n_zones)) else 'INCOMPLETA!'} "
          f"({len(covered)}/{n_zones} zonas)")

    print()
    print(f"  {'Antena':<8} {'Ativa?':<7} {'Custo':>6}  Zonas cobertas")
    print("  " + "-" * 55)
    for j in range(n_sites):
        ativa = "SIM" if j in selected else "nao"
        cost_str = f"{costs[j]:.1f}" if j in selected else ""
        zones_str = ", ".join(zone_names[i] for i in sorted(coverage[j]))
        print(f"  {site_names[j]:<8} {ativa:<7} {cost_str:>6}  {zones_str}")
    print("=" * 65)


# ------------------------------------------------------------------
# Sensitivity analysis: vary cost of one site
# ------------------------------------------------------------------

def sensitivity_analysis(
    cost_range: list[float],
    site_idx: int = 0,
    costs_base: list[float] = DEFAULT_COSTS,
    coverage: list[frozenset] = DEFAULT_COVERAGE,
    n_zones: int = N_ZONES,
) -> list[tuple[float, float | None, list[int] | None]]:
    """
    Vary cost of site `site_idx` over `cost_range`.
    Returns list of (cost_val, optimal_total_cost, selected_sites).
    """
    results = []
    for c_val in cost_range:
        costs = list(costs_base)
        costs[site_idx] = c_val
        model = build_model(costs, coverage, n_zones)
        solver = BranchAndBound(model, strategy="best_first",
                                branching="most_infeasible", verbose=False)
        r = solver.solve()
        sel = decode_solution(r, len(costs)) if r.x is not None else None
        results.append((c_val, r.obj_value, sel))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("PROBLEMA 6 - Cobertura de Zonas de Rede (Set Covering PIB)")
    print("=" * 65)
    print(f"\n{N_ZONES} zonas a cobrir, {N_SITES} locais candidatos:\n")
    print(f"  {'Antena':<8} {'Custo':>7}  Zonas cobertas")
    print("  " + "-" * 42)
    for j in range(N_SITES):
        zones_str = ", ".join(ZONE_NAMES[i] for i in sorted(DEFAULT_COVERAGE[j]))
        print(f"  {SITE_NAMES[j]:<8} {DEFAULT_COSTS[j]:>7.1f}  {zones_str}")

    model = build_model(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES, SITE_NAMES)

    greedy_sel, greedy_cost = greedy_set_cover(DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES)
    x_greedy = greedy_to_x_vector(greedy_sel, N_SITES)
    print(f"\nHeuristica Greedy: {{{', '.join(SITE_NAMES[j] for j in greedy_sel)}}} "
          f"| custo = {greedy_cost:.1f}")

    print("\n--- Branch-and-Bound ---")
    bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible",
                        initial_incumbent=greedy_cost, initial_x=x_greedy)
    r_bb = bb.solve()
    print_solution(r_bb, DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES,
                   label="B&B | Instancia original",
                   site_names=SITE_NAMES, zone_names=ZONE_NAMES)

    print("\n--- Branch-and-Cut (cortes de Gomory) ---")
    bc = BranchAndCut(model, strategy="best_first", branching="most_infeasible",
                      cut_types=["gomory"],
                      initial_incumbent=greedy_cost, initial_x=x_greedy)
    r_bc = bc.solve()
    print_solution(r_bc, DEFAULT_COSTS, DEFAULT_COVERAGE, N_ZONES,
                   label="B&C | Instancia original",
                   site_names=SITE_NAMES, zone_names=ZONE_NAMES)

    print(f"\nB&B nos={r_bb.nodes_explored} | B&C nos={r_bc.nodes_explored}")

    # ---- Sensitivity: vary cost of A1 ----
    print("\n" + "=" * 65)
    print("ANALISE DE SENSIBILIDADE — custo da antena A1")
    print("=" * 65)
    cost_range = [float(v) for v in range(1, 11)]
    sens = sensitivity_analysis(cost_range, site_idx=0)

    print(f"\n  {'Custo A1':>10} {'Custo otimo':>12}  Solucao")
    print("  " + "-" * 55)
    prev_sol = None
    for c_val, total, selected in sens:
        if total is None:
            print(f"  {c_val:>10.1f} {'inviavel':>12}")
            continue
        sol_str = "{" + ", ".join(SITE_NAMES[j] for j in (selected or [])) + "}"
        change = " <-- mudanca" if selected != prev_sol and prev_sol is not None else ""
        print(f"  {c_val:>10.1f} {total:>12.1f}  {sol_str}{change}")
        prev_sol = selected

    print("\nInterpretacao:")
    print("  c(A1) <= 5 -> otimo = {A1, A2, A6}, custo = c(A1) + 7")
    print("  c(A1)  = 6 -> empate: {A1,A2,A6} = {A3,A4,A6} = 13")
    print("  c(A1) >= 7 -> otimo = {A3, A4, A6}, custo = 13 (fixo)")

    # ---- Random instances ----
    print("\n" + "=" * 65)
    print("INSTANCIAS ALEATORIAS")
    print("=" * 65)
    print(f"\n  {'Instancia':<28} {'Greedy':>8} {'B&B':>8} {'B&B nos':>9} "
          f"{'B&C':>8} {'B&C nos':>9} {'Reducao':>9}")
    print("  " + "-" * 83)

    for n_z, n_s, seed in [(10, 8, 1), (15, 10, 2), (20, 12, 3)]:
        costs_r, cov_r = random_instance(n_z, n_s, zones_per_site=4, seed=seed)
        sn_r = [f"S{j+1}" for j in range(n_s)]

        g_sel, g_cost = greedy_set_cover(costs_r, cov_r, n_z)
        x_g = greedy_to_x_vector(g_sel, n_s)

        model_r = build_model(costs_r, cov_r, n_z, sn_r)

        r_bb_r = BranchAndBound(model_r, strategy="best_first",
                                branching="most_infeasible",
                                initial_incumbent=g_cost, initial_x=x_g,
                                time_limit=30.0).solve()
        r_bc_r = BranchAndCut(model_r, strategy="best_first",
                               branching="most_infeasible",
                               cut_types=["gomory"],
                               initial_incumbent=g_cost, initial_x=x_g,
                               time_limit=30.0).solve()

        bb_v = f"{r_bb_r.obj_value:.1f}" if r_bb_r.obj_value is not None else "?"
        bc_v = f"{r_bc_r.obj_value:.1f}" if r_bc_r.obj_value is not None else "?"
        if r_bb_r.nodes_explored > 0 and r_bc_r.nodes_explored > 0 and r_bb_r.nodes_explored > 1:
            red = f"{100*(1 - r_bc_r.nodes_explored/r_bb_r.nodes_explored):.1f}%"
        else:
            red = "N/A"
        label = f"({n_z} zonas, {n_s} antenas, s={seed})"
        print(f"  {label:<28} {g_cost:>8.1f} {bb_v:>8} {r_bb_r.nodes_explored:>9} "
              f"{bc_v:>8} {r_bc_r.nodes_explored:>9} {red:>9}")
