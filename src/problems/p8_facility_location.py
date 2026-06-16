"""
Problema 8 - Localizacao de Servidores CDN [PLIM]

Uma empresa de CDN precisa decidir quais centros de dados instalar
e a qual centro cada regiao sera atribuida, minimizando custo de
instalacao + custo de servico (latencia).

Formulacao PLIM (Uncapacitated Facility Location - UFL):
    min  sum_j f_j * y_j  +  sum_i sum_j c_{ij} * x_{ij}
    s.t.
    (1)  sum_j x_{ij} = 1             para cada regiao i     (atribuicao unica)
    (2)  x_{ij} <= y_j                para cada i,j          (so atende se instalado)
    (3)  sum_j y_j >= 1                                       (ao menos 1 centro)
    (4)  sum_j y_j <= MAX_OPEN                                (no maximo MAX_OPEN centros)
    (5)  sum_j f_j * y_j <= BUDGET                           (orcamento)
    (6)  y_j in {0,1},  x_{ij} in [0,1]                     (PLIM: x continuo)

Variaveis (indexacao flat: n_f + n_c*n_f variaveis):
    y_j     -> j                  (j = 0..n_f-1)
    x_{ij}  -> n_f + i*n_f + j   (i = 0..n_c-1, j = 0..n_f-1)

Instancia (PDF): 4 centros candidatos, 6 regioes
    Centros: C1(f=30), C2(f=25), C3(f=35), C4(f=20)
    MAX_OPEN = 3, BUDGET = 120

    Custo de servico (latencia ms):
           C1  C2  C3  C4
    R1  [  3,  8,  6,  9 ]
    R2  [  7,  2,  5,  4 ]
    R3  [  5,  6,  2,  7 ]
    R4  [  9,  3,  8,  2 ]
    R5  [  4,  7,  3,  8 ]
    R6  [  6,  5,  7,  3 ]

Otimo: {C4}, custo = 20 + (9+4+7+2+8+3) = 20 + 33 = 53.
Greedy (melhor centro unico): {C4}, custo = 53.

Analise de sensibilidade (custo de C4 em [10..50]):
    f(C4) <  23 -> {C4} otimo, custo = f(C4) + 33
    f(C4) =  23 -> empate: {C4} = {C2} = 56
    f(C4) >  23 -> {C2} otimo, custo = 56 (fixo: 25 + 31)
"""
from __future__ import annotations
import random
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult


# ------------------------------------------------------------------
# Default instance (PDF)
# ------------------------------------------------------------------

N_FACILITIES = 4
N_CLIENTS = 6
MAX_OPEN = 3
BUDGET = 120.0

FACILITY_NAMES = ["C1", "C2", "C3", "C4"]
CLIENT_NAMES   = ["R1", "R2", "R3", "R4", "R5", "R6"]

DEFAULT_FIXED_COSTS: list[float] = [30.0, 25.0, 35.0, 20.0]

# service_costs[i][j] = cost to serve client i from facility j
DEFAULT_SERVICE_COSTS: list[list[float]] = [
    [3, 8, 6, 9],   # R1
    [7, 2, 5, 4],   # R2
    [5, 6, 2, 7],   # R3
    [9, 3, 8, 2],   # R4
    [4, 7, 3, 8],   # R5
    [6, 5, 7, 3],   # R6
]

# Alias for backward compatibility
DEFAULT_LATENCY = DEFAULT_SERVICE_COSTS


# ------------------------------------------------------------------
# Variable index helpers
# ------------------------------------------------------------------

def iy(j: int, n_f: int) -> int:
    """Index of binary installation variable y_j."""
    return j

def ix(i: int, j: int, n_f: int) -> int:
    """Index of continuous allocation variable x_{ij}."""
    return n_f + i * n_f + j


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    fixed_costs: list[float],
    service_costs: list[list[float]],
    n_f: int | None = None,
    n_c: int | None = None,
    facility_names: list[str] | None = None,
    client_names: list[str] | None = None,
    max_open: int | None = MAX_OPEN,
    budget: float | None = BUDGET,
) -> MIPModel:
    """
    Build Uncapacitated Facility Location MIPModel with budget and max-open constraints.

    y_j binary, x_{ij} continuous in [0,1].
    """
    if n_f is None:
        n_f = len(fixed_costs)
    if n_c is None:
        n_c = len(service_costs)
    if facility_names is None:
        facility_names = [f"s{j+1}" for j in range(n_f)]
    if client_names is None:
        client_names = [f"c{i+1}" for i in range(n_c)]

    n_vars = n_f + n_c * n_f

    # Objective: min sum_j f_j*y_j + sum_{i,j} c_{ij}*x_{ij}
    c = [0.0] * n_vars
    for j in range(n_f):
        c[iy(j, n_f)] = float(fixed_costs[j])
    for i in range(n_c):
        for j in range(n_f):
            c[ix(i, j, n_f)] = float(service_costs[i][j])

    # Bounds: y_j in [0,1] (binary), x_{ij} in [0,1] (continuous)
    bounds = [(0.0, 1.0)] * n_vars

    # Integrality: y binary (2), x continuous (0)
    integrality = [0] * n_vars
    for j in range(n_f):
        integrality[iy(j, n_f)] = 2

    A_ub, b_ub = [], []

    # (2) Linking: x_{ij} - y_j <= 0
    for i in range(n_c):
        for j in range(n_f):
            row = [0.0] * n_vars
            row[ix(i, j, n_f)] =  1.0
            row[iy(j, n_f)]    = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # (3) At least 1 facility: -sum_j y_j <= -1
    row = [0.0] * n_vars
    for j in range(n_f):
        row[iy(j, n_f)] = -1.0
    A_ub.append(row)
    b_ub.append(-1.0)

    # (4) At most MAX_OPEN facilities: sum_j y_j <= max_open
    if max_open is not None:
        row = [0.0] * n_vars
        for j in range(n_f):
            row[iy(j, n_f)] = 1.0
        A_ub.append(row)
        b_ub.append(float(max_open))

    # (5) Budget: sum_j f_j * y_j <= budget
    if budget is not None:
        row = [0.0] * n_vars
        for j in range(n_f):
            row[iy(j, n_f)] = float(fixed_costs[j])
        A_ub.append(row)
        b_ub.append(float(budget))

    # (1) Assignment: sum_j x_{ij} = 1
    A_eq, b_eq = [], []
    for i in range(n_c):
        row = [0.0] * n_vars
        for j in range(n_f):
            row[ix(i, j, n_f)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    var_names  = list(facility_names)
    var_names += [f"x{client_names[i]}{facility_names[j]}"
                  for i in range(n_c) for j in range(n_f)]

    return MIPModel(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, integrality=integrality,
        sense="min",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# Greedy heuristic: best single-facility solution
# ------------------------------------------------------------------

def greedy_solution(
    fixed_costs: list[float],
    service_costs: list[list[float]],
    n_f: int | None = None,
    n_c: int | None = None,
) -> tuple[float, np.ndarray]:
    """
    Upper bound: open each single facility and keep the best.
    Returns (total_cost, x_vector).
    """
    if n_f is None:
        n_f = len(fixed_costs)
    if n_c is None:
        n_c = len(service_costs)

    best_cost = float("inf")
    best_x: np.ndarray = np.zeros(n_f + n_c * n_f)

    for j in range(n_f):
        cost = fixed_costs[j] + sum(service_costs[i][j] for i in range(n_c))
        if cost < best_cost:
            best_cost = cost
            x = np.zeros(n_f + n_c * n_f)
            x[iy(j, n_f)] = 1.0
            for i in range(n_c):
                x[ix(i, j, n_f)] = 1.0
            best_x = x

    return best_cost, best_x


# ------------------------------------------------------------------
# Solution decoder and printer
# ------------------------------------------------------------------

def decode_solution(
    result: BBResult,
    n_f: int,
    n_c: int,
) -> tuple[list[int], list[int]]:
    """
    Returns (open_f, assignments) where:
        open_f[j]       = 1 if facility j is installed
        assignments[i]  = j  (client i assigned to facility j)
    """
    if result.x is None:
        return [], []
    open_f = [round(result.x[iy(j, n_f)]) for j in range(n_f)]
    assignments = [
        max(range(n_f), key=lambda j: result.x[ix(i, j, n_f)])
        for i in range(n_c)
    ]
    return open_f, assignments


def print_solution(
    result: BBResult,
    fixed_costs: list[float],
    service_costs: list[list[float]],
    label: str = "",
    facility_names: list[str] | None = None,
    client_names: list[str] | None = None,
):
    n_f = len(fixed_costs)
    n_c = len(service_costs)
    if facility_names is None:
        facility_names = [f"C{j+1}" for j in range(n_f)]
    if client_names is None:
        client_names = [f"R{i+1}" for i in range(n_c)]

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

    open_f, assignments = decode_solution(result, n_f, n_c)
    total_fixed   = sum(fixed_costs[j] * open_f[j] for j in range(n_f))
    total_service = sum(service_costs[i][assignments[i]] for i in range(n_c))

    print(f"Custo total     : {result.obj_value:.1f}")
    print(f"  Instalacao    : {total_fixed:.1f}")
    print(f"  Servico       : {total_service:.1f}")
    installed = [facility_names[j] for j in range(n_f) if open_f[j]]
    print(f"Centros abertos : {{{', '.join(installed)}}}")

    print()
    print(f"  {'Regiao':<8} {'Centro':>10} {'Custo serv.':>12}")
    print("  " + "-" * 32)
    for i in range(n_c):
        j = assignments[i]
        print(f"  {client_names[i]:<8} {facility_names[j]:>10} {service_costs[i][j]:>12.1f}")
    print("=" * 65)


# ------------------------------------------------------------------
# Random instance generator
# ------------------------------------------------------------------

def random_instance(
    n_f: int,
    n_c: int,
    max_fixed: float = 100.0,
    max_service: float = 20.0,
    seed: int = 0,
) -> tuple[list[float], list[list[float]]]:
    rng = random.Random(seed)
    fixed = [round(rng.uniform(10, max_fixed), 1) for _ in range(n_f)]
    service = [[round(rng.uniform(1, max_service), 1) for _ in range(n_f)]
               for _ in range(n_c)]
    return fixed, service


# ------------------------------------------------------------------
# Sensitivity analysis: vary fixed cost of C4 (facility index 3)
# ------------------------------------------------------------------

def sensitivity_analysis(
    f4_range: list[float],
    fixed_costs_base: list[float] = DEFAULT_FIXED_COSTS,
    service_costs: list[list[float]] = DEFAULT_SERVICE_COSTS,
) -> list[tuple[float, float | None, list[int] | None]]:
    """
    Vary fixed cost of facility 3 (C4) over f4_range.
    Returns list of (f4, optimal_cost, open_facilities).
    """
    n_f = len(fixed_costs_base)
    n_c = len(service_costs)
    results = []
    for f4 in f4_range:
        fc = list(fixed_costs_base)
        fc[3] = f4
        model = build_model(fc, service_costs, n_f, n_c)
        g_cost, g_x = greedy_solution(fc, service_costs, n_f, n_c)
        solver = BranchAndBound(model, strategy="best_first",
                                branching="most_infeasible",
                                initial_incumbent=g_cost, initial_x=g_x,
                                verbose=False)
        r = solver.solve()
        open_f, _ = decode_solution(r, n_f, n_c) if r.x is not None else (None, None)
        results.append((f4, r.obj_value, open_f))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    fn = FACILITY_NAMES
    cn = CLIENT_NAMES

    print("=" * 65)
    print("PROBLEMA 8 - Localizacao de Servidores CDN (PLIM - UFL)")
    print("=" * 65)
    print(f"\n{N_FACILITIES} centros candidatos, {N_CLIENTS} regioes | "
          f"Max abertos: {MAX_OPEN} | Orcamento: {BUDGET:.0f}\n")

    print("  Custo de instalacao:")
    for j in range(N_FACILITIES):
        print(f"    {fn[j]}: ${DEFAULT_FIXED_COSTS[j]:.0f}")

    print("\n  Custo de servico (latencia ms):")
    print("         " + "  ".join(f"{fn[j]:>5}" for j in range(N_FACILITIES)))
    for i in range(N_CLIENTS):
        row = "  ".join(f"{DEFAULT_SERVICE_COSTS[i][j]:>5.0f}" for j in range(N_FACILITIES))
        print(f"  {cn[i]}:    {row}")

    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS,
                        N_FACILITIES, N_CLIENTS, fn, cn)

    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS)
    print(f"\nHeuristica (melhor centro unico): custo = {g_cost:.1f}")

    print("\n--- Branch-and-Bound ---")
    bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible",
                        initial_incumbent=g_cost, initial_x=g_x)
    r_bb = bb.solve()
    print_solution(r_bb, DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS,
                   label="B&B | Instancia original",
                   facility_names=fn, client_names=cn)

    print("\n--- Branch-and-Cut (cortes de Gomory em y_j fracionarios) ---")
    bc = BranchAndCut(model, strategy="best_first", branching="most_infeasible",
                      cut_types=["gomory"],
                      initial_incumbent=g_cost, initial_x=g_x)
    r_bc = bc.solve()
    print_solution(r_bc, DEFAULT_FIXED_COSTS, DEFAULT_SERVICE_COSTS,
                   label="B&C | Instancia original",
                   facility_names=fn, client_names=cn)

    print(f"\nB&B nos={r_bb.nodes_explored} | B&C nos={r_bc.nodes_explored}")

    # ---- Sensitivity: vary fixed cost of C4 ----
    print("\n" + "=" * 65)
    print("ANALISE DE SENSIBILIDADE — custo de instalacao de C4")
    print("=" * 65)
    f4_range = [10.0, 15.0, 20.0, 23.0, 25.0, 30.0, 40.0, 50.0]
    sens = sensitivity_analysis(f4_range)

    print(f"\n  {'Custo C4':>10} {'Custo total':>12}  Centros abertos")
    print("  " + "-" * 50)
    prev_open = None
    for f4, total, open_f in sens:
        if total is None:
            print(f"  {f4:>10.0f} {'inviavel':>12}")
            continue
        open_str = "{" + ", ".join(fn[j] for j in range(N_FACILITIES) if open_f[j]) + "}"
        change = " <-- mudanca" if open_f != prev_open and prev_open is not None else ""
        print(f"  {f4:>10.0f} {total:>12.1f}  {open_str}{change}")
        prev_open = open_f

    print("\nInterpretacao:")
    print("  f(C4) <  23 -> {C4} otimo: custo = f(C4) + 33 < 56")
    print("  f(C4) =  23 -> empate: {C4}=56 = {C2}=56")
    print("  f(C4) >  23 -> {C2} otimo: custo = 56 (fixo: 25 + 31)")

    # ---- Random instances ----
    print("\n" + "=" * 65)
    print("INSTANCIAS ALEATORIAS")
    print("=" * 65)
    print(f"\n  {'Instancia':<25} {'Greedy':>8} {'B&B':>8} {'B&B nos':>9} "
          f"{'B&C':>8} {'B&C nos':>9}")
    print("  " + "-" * 72)

    for nf, nc, seed in [(4, 10, 1), (5, 15, 2), (6, 20, 3)]:
        fc_r, sc_r = random_instance(nf, nc, seed=seed)
        fn_r = [f"C{j+1}" for j in range(nf)]
        cn_r = [f"R{i+1}" for i in range(nc)]

        g_cost_r, g_x_r = greedy_solution(fc_r, sc_r, nf, nc)
        model_r = build_model(fc_r, sc_r, nf, nc, fn_r, cn_r)

        r_bb_r = BranchAndBound(model_r, strategy="best_first",
                                branching="most_infeasible",
                                initial_incumbent=g_cost_r, initial_x=g_x_r,
                                time_limit=30.0, verbose=False).solve()
        r_bc_r = BranchAndCut(model_r, strategy="best_first",
                               branching="most_infeasible",
                               cut_types=["gomory"],
                               initial_incumbent=g_cost_r, initial_x=g_x_r,
                               time_limit=30.0).solve()

        bb_v = f"{r_bb_r.obj_value:.1f}" if r_bb_r.obj_value is not None else "?"
        bc_v = f"{r_bc_r.obj_value:.1f}" if r_bc_r.obj_value is not None else "?"
        label = f"({nf} centros, {nc} regioes, s={seed})"
        print(f"  {label:<25} {g_cost_r:>8.1f} {bb_v:>8} {r_bb_r.nodes_explored:>9} "
              f"{bc_v:>8} {r_bc_r.nodes_explored:>9}")
