"""
Problema 8 - Localizacao de Servidores CDN [PLIM]

Uma empresa de CDN (Content Delivery Network) precisa decidir quais
servidores instalar e a qual servidor cada cliente sera atribuido,
minimizando custo de instalacao + custo de latencia de atendimento.

Formulacao PLIM (Uncapacitated Facility Location - UFL):
    min  sum_j f_j * y_j  +  sum_i sum_j c_{ij} * x_{ij}
    s.t.
    (1)  sum_j x_{ij} = 1          para cada cliente i     (atribuicao unica)
    (2)  x_{ij} <= y_j             para cada i,j           (so atende se instalado)
    (3)  y_j in {0,1},  x_{ij} in [0,1]                   (PLIM: x continuo)

Variaveis (indexacao flat: n_f + n_c*n_f variaveis):
    y_j     -> j                  (j = 0..n_f-1)
    x_{ij}  -> n_f + i*n_f + j   (i = 0..n_c-1, j = 0..n_f-1)

Instancia (3 servidores candidatos, 4 clientes):
    Servidores: S1(custo=60), S2(custo=30), S3(custo=variable, default=60)
    Clientes:   C1, C2, C3, C4

    Latencia (ms):
           S1  S2  S3
    C1  [  5, 50, 62 ]   (C1 prefere S1)
    C2  [ 55,  5, 58 ]   (C2 prefere S2)
    C3  [ 52, 45,  6 ]   (C3 prefere S3)
    C4  [  8, 48,  7 ]   (C4 quase igual entre S1 e S3)

Otimo (custo S3=60): instalar S1+S2, custo = 60+30 + (5+5+45+8) = 153.
  C1->S1(5), C2->S2(5), C3->S2(45), C4->S1(8).

Analise de sensibilidade (custo de instalacao de S3 em [20..80]):
    custo S3 < 55  -> instalar S2+S3 (C3,C4 migram para S3), custo = custo_S3 + 98
    custo S3 = 55  -> empate (custo 153): S1+S2 = S2+S3
    custo S3 > 55  -> instalar S1+S2 (custo = 153, constante)
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult


# ------------------------------------------------------------------
# Default instance
# ------------------------------------------------------------------

N_FACILITIES = 3
N_CLIENTS = 4

FACILITY_NAMES = [f"S{j+1}" for j in range(N_FACILITIES)]
CLIENT_NAMES   = [f"C{i+1}" for i in range(N_CLIENTS)]

DEFAULT_FIXED_COSTS: list[float] = [60.0, 30.0, 60.0]   # installation cost per server

# latency[i][j] = cost to serve client i from facility j
DEFAULT_LATENCY: list[list[float]] = [
    [ 5, 50, 62],   # C1: best at S1
    [55,  5, 58],   # C2: best at S2
    [52, 45,  6],   # C3: best at S3
    [ 8, 48,  7],   # C4: near-tie S1 and S3
]


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
    latency: list[list[float]],
    n_f: int | None = None,
    n_c: int | None = None,
    facility_names: list[str] | None = None,
    client_names: list[str] | None = None,
) -> MIPModel:
    """
    Build Uncapacitated Facility Location MIPModel.

    y_j binary, x_{ij} continuous in [0,1].
    """
    if n_f is None:
        n_f = len(fixed_costs)
    if n_c is None:
        n_c = len(latency)
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
            c[ix(i, j, n_f)] = float(latency[i][j])

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
    latency: list[list[float]],
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
        n_c = len(latency)

    best_cost = float("inf")
    best_x: np.ndarray = np.zeros(n_f + n_c * n_f)

    for j in range(n_f):
        cost = fixed_costs[j] + sum(latency[i][j] for i in range(n_c))
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
    latency: list[list[float]],
    label: str = "",
    facility_names: list[str] | None = None,
    client_names: list[str] | None = None,
):
    n_f = len(fixed_costs)
    n_c = len(latency)
    if facility_names is None:
        facility_names = [f"S{j+1}" for j in range(n_f)]
    if client_names is None:
        client_names = [f"C{i+1}" for i in range(n_c)]

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
    total_latency = sum(latency[i][assignments[i]] for i in range(n_c))

    print(f"Custo total     : {result.obj_value:.1f}")
    print(f"  Instalacao    : {total_fixed:.1f}")
    print(f"  Latencia      : {total_latency:.1f}")
    installed = [facility_names[j] for j in range(n_f) if open_f[j]]
    print(f"Servidores abertos: {{{', '.join(installed)}}}")

    print()
    print(f"  {'Cliente':<8} {'Servidor':>10} {'Latencia':>10}")
    print("  " + "-" * 30)
    for i in range(n_c):
        j = assignments[i]
        print(f"  {client_names[i]:<8} {facility_names[j]:>10} {latency[i][j]:>10.1f}")
    print("=" * 65)


# ------------------------------------------------------------------
# Sensitivity analysis: vary fixed cost of S3 (facility index 2)
# ------------------------------------------------------------------

def sensitivity_analysis(
    f3_range: list[float],
    fixed_costs_base: list[float] = DEFAULT_FIXED_COSTS,
    latency: list[list[float]] = DEFAULT_LATENCY,
) -> list[tuple[float, float | None, list[int] | None]]:
    """
    Vary fixed cost of facility 2 (S3) over f3_range.
    Returns list of (f3, optimal_cost, open_facilities).
    """
    n_f = len(fixed_costs_base)
    n_c = len(latency)
    results = []
    for f3 in f3_range:
        fc = list(fixed_costs_base)
        fc[2] = f3
        model = build_model(fc, latency, n_f, n_c)
        g_cost, g_x = greedy_solution(fc, latency, n_f, n_c)
        solver = BranchAndBound(model, strategy="best_first",
                                branching="most_infeasible",
                                initial_incumbent=g_cost, initial_x=g_x,
                                verbose=False)
        r = solver.solve()
        open_f, _ = decode_solution(r, n_f, n_c) if r.x is not None else (None, None)
        results.append((f3, r.obj_value, open_f))
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
    print(f"\n{N_FACILITIES} servidores candidatos, {N_CLIENTS} clientes\n")

    print("  Custo de instalacao:")
    for j in range(N_FACILITIES):
        print(f"    {fn[j]}: ${DEFAULT_FIXED_COSTS[j]:.0f}")

    print("\n  Latencia (ms):")
    print("         " + "  ".join(f"{fn[j]:>5}" for j in range(N_FACILITIES)))
    for i in range(N_CLIENTS):
        row = "  ".join(f"{DEFAULT_LATENCY[i][j]:>5.0f}" for j in range(N_FACILITIES))
        print(f"  {cn[i]}:    {row}")

    model = build_model(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY, N_FACILITIES, N_CLIENTS, fn, cn)

    g_cost, g_x = greedy_solution(DEFAULT_FIXED_COSTS, DEFAULT_LATENCY)
    print(f"\nHeuristica (melhor servidor unico): custo = {g_cost:.1f}")

    print("\n--- Branch-and-Bound ---")
    bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible",
                        initial_incumbent=g_cost, initial_x=g_x)
    r_bb = bb.solve()
    print_solution(r_bb, DEFAULT_FIXED_COSTS, DEFAULT_LATENCY,
                   label="B&B | Instancia original",
                   facility_names=fn, client_names=cn)

    print("\n--- Branch-and-Cut (cortes de Gomory em y_j fracionarios) ---")
    bc = BranchAndCut(model, strategy="best_first", branching="most_infeasible",
                      cut_types=["gomory"],
                      initial_incumbent=g_cost, initial_x=g_x)
    r_bc = bc.solve()
    print_solution(r_bc, DEFAULT_FIXED_COSTS, DEFAULT_LATENCY,
                   label="B&C | Instancia original",
                   facility_names=fn, client_names=cn)

    print(f"\nB&B nos={r_bb.nodes_explored} | B&C nos={r_bc.nodes_explored}")

    # ---- Sensitivity: vary fixed cost of S3 ----
    print("\n" + "=" * 65)
    print("ANALISE DE SENSIBILIDADE — custo de instalacao de S3")
    print("=" * 65)
    f3_range = [float(v) for v in range(20, 81, 10)]
    sens = sensitivity_analysis(f3_range)

    print(f"\n  {'Custo S3':>10} {'Custo total':>12}  Servidores abertos")
    print("  " + "-" * 50)
    prev_open = None
    for f3, total, open_f in sens:
        if total is None:
            print(f"  {f3:>10.0f} {'inviavel':>12}")
            continue
        open_str = "{" + ", ".join(fn[j] for j in range(N_FACILITIES) if open_f[j]) + "}"
        change = " <-- mudanca" if open_f != prev_open and prev_open is not None else ""
        print(f"  {f3:>10.0f} {total:>12.1f}  {open_str}{change}")
        prev_open = open_f

    print("\nInterpretacao:")
    print("  Custo S3 < 55  -> instalar S2+S3: C3,C4 migram para S3 (6ms,7ms)")
    print("  Custo S3 = 55  -> empate (custo 153): {S1,S2} = {S2,S3}")
    print("  Custo S3 > 55  -> instalar S1+S2: custo fixo = 153 (constante)")
