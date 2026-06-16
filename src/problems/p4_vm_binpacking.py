"""
Problema 4 - Escalonamento de Maquinas Virtuais em Nuvem [PLIP]

Bin Packing: alocar n VMs em servidores fisicos com capacidade C,
minimizando o numero de servidores utilizados.

Variaveis:
    y[j] in {0,1} : servidor j esta ativo     (j = 0..n-1)
    x[i][j] in {0,1} : VM i no servidor j     (i,j = 0..n-1)

Formulacao:
    min  sum_j y[j]
    s.t.
    (1) sum_i d[i] * x[i][j] <= C * y[j]   para cada j  (capacidade)
    (2) sum_j x[i][j] = 1                   para cada i  (atribuicao unica)
    (3) x[i][j] <= y[j]                     para cada i,j  (ligacao VM-servidor)
    (4) y[j+1] <= y[j]                      para j=0..n-2  (quebra de simetria)
        y[j], x[i][j] in {0,1}

Indice das variaveis:
    y[j]    -> j            (j = 0..n-1)
    x[i][j] -> n + i*n + j  (i,j = 0..n-1)
"""
from __future__ import annotations
import math
import random
import time
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult


def print_cut_log(bc: BranchAndCut, label: str = "") -> None:
    """Print a readable summary of all cuts generated during a B&C run."""
    prefix = f"  [{label}] " if label else "  "
    log = getattr(bc, "_cut_log", [])
    if not log:
        print(f"{prefix}Nenhum corte adicionado.")
        return
    by_type: dict[str, int] = {}
    for entry in log:
        by_type[entry["type"]] = by_type.get(entry["type"], 0) + 1
    total = len(log)
    summary = ", ".join(f"{t}={cnt}" for t, cnt in by_type.items())
    print(f"{prefix}{total} corte(s) adicionado(s): {summary}")
    for i, entry in enumerate(log, 1):
        print(f"    Corte {i:3d}: tipo={entry['type']:<8} no={entry['node']:>4}  rhs={entry['rhs']:.4f}")


# ------------------------------------------------------------------
# Lower bound: LP relaxation of bin packing = ceil(sum_d / C)
# ------------------------------------------------------------------

def lb_fractional(demands: list[float], capacity: float) -> int:
    return math.ceil(sum(demands) / capacity)


# ------------------------------------------------------------------
# Upper bound heuristic: First-Fit Decreasing (FFD)
# ------------------------------------------------------------------

def first_fit_decreasing(
    demands: list[float],
    capacity: float,
) -> tuple[int, list[list[int]]]:
    """
    Returns (n_bins_used, bins) where bins[j] = list of VM indices in bin j.
    """
    order = sorted(range(len(demands)), key=lambda i: demands[i], reverse=True)
    bins: list[tuple[float, list[int]]] = []  # (current_load, vm_list)

    for vm in order:
        d = demands[vm]
        placed = False
        for b in range(len(bins)):
            load, vms = bins[b]
            if load + d <= capacity + 1e-9:
                bins[b] = (load + d, vms + [vm])
                placed = True
                break
        if not placed:
            bins.append((d, [vm]))

    return len(bins), [vms for _, vms in bins]


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(demands: list[float], capacity: float, max_servers: int | None = None) -> MIPModel:
    """
    Build bin-packing MIPModel.

    max_servers: number of potential server slots (default: len(demands)).
                 Pass the FFD result to keep the model compact for large n.
    """
    n = len(demands)          # number of VMs
    k = max_servers or n      # number of potential server slots
    n_vars = k + n * k

    def iy(j):    return j
    def ix(i, j): return k + i * k + j

    # Objective: minimize sum y_j
    c = [0.0] * n_vars
    for j in range(k):
        c[iy(j)] = 1.0

    bounds = [(0.0, 1.0)] * n_vars
    integrality = [2] * n_vars

    A_ub, b_ub = [], []

    # Objective lower bound: sum_j y_j >= ceil(sum_i d_i / C)
    # Valid cut: optimal bins >= fractional relaxation rounded up.
    # When lb == FFD == warm-start incumbent, LP at root = lb = incumbent -> root pruned in 1 node.
    lb_obj = math.ceil(sum(demands) / capacity)
    row = [0.0] * n_vars
    for j in range(k):
        row[iy(j)] = -1.0   # -sum y_j <= -lb  <=>  sum y_j >= lb
    A_ub.append(row)
    b_ub.append(-float(lb_obj))

    # (1) Capacity: sum_i d_i * x_ij - C * y_j <= 0
    for j in range(k):
        row = [0.0] * n_vars
        row[iy(j)] = -float(capacity)
        for i in range(n):
            row[ix(i, j)] = float(demands[i])
        A_ub.append(row)
        b_ub.append(0.0)

    # (3) Linking: x_ij - y_j <= 0
    for i in range(n):
        for j in range(k):
            row = [0.0] * n_vars
            row[iy(j)] = -1.0
            row[ix(i, j)] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # (4) Symmetry breaking: y_{j+1} - y_j <= 0
    for j in range(k - 1):
        row = [0.0] * n_vars
        row[iy(j + 1)] = 1.0
        row[iy(j)] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Relaxed capacity (for cover cuts): sum_i d_i * x_ij <= C
    # Valid since x_ij <= y_j <= 1; allows cover_cuts to find knapsack covers.
    for j in range(k):
        row = [0.0] * n_vars
        for i in range(n):
            row[ix(i, j)] = float(demands[i])
        A_ub.append(row)
        b_ub.append(float(capacity))

    # (2) Assignment: sum_j x_ij = 1
    A_eq, b_eq = [], []
    for i in range(n):
        row = [0.0] * n_vars
        for j in range(k):
            row[ix(i, j)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    var_names = [f"y{j+1}" for j in range(k)]
    var_names += [f"x{i+1}{j+1}" for i in range(n) for j in range(k)]

    return MIPModel(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, integrality=integrality,
        sense="min",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# Encode FFD solution as an x-vector for warm-starting B&B
# ------------------------------------------------------------------

def ffd_to_x_vector(bins: list[list[int]], n: int, k: int) -> np.ndarray:
    """
    Convert FFD bin assignment into the MIPModel variable vector.
    bins[j] = list of 0-indexed VM indices assigned to bin j.
    n = number of VMs, k = max_servers (= len(bins)).
    """
    x = np.zeros(k + n * k)
    for j, vms in enumerate(bins):
        x[j] = 1.0                  # y[j] = 1
        for i in vms:
            x[k + i * k + j] = 1.0  # x[i][j] = 1
    return x


# ------------------------------------------------------------------
# Solution decoder
# ------------------------------------------------------------------

def decode_solution(
    result: BBResult,
    demands: list[float],
    capacity: float,
    max_servers: int | None = None,
) -> list[list[int]] | None:
    """Returns bins[j] = list of VM indices, or None if no solution."""
    if result.x is None:
        return None
    n = len(demands)
    k = max_servers or n
    bins: dict[int, list[int]] = {}
    for i in range(n):
        for j in range(k):
            if round(result.x[k + i * k + j]) == 1:
                bins.setdefault(j, []).append(i)
    used = sorted(j for j in range(k) if round(result.x[j]) == 1)
    return [bins.get(j, []) for j in used]


# ------------------------------------------------------------------
# Printer
# ------------------------------------------------------------------

def print_solution(
    result: BBResult,
    demands: list[float],
    capacity: float,
    label: str = "",
    max_servers: int | None = None,
):
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

    n_bins = round(result.obj_value)
    print(f"Servidores usados: {n_bins}")
    bins = decode_solution(result, demands, capacity, max_servers)
    if bins:
        for j, vms in enumerate(bins):
            load = sum(demands[i] for i in vms)
            vm_str = ", ".join(f"VM{i+1}(d={demands[i]:.0f})" for i in vms)
            print(f"  Bin {j+1}: [{vm_str}] -> carga={load:.0f}/{capacity:.0f}")
    print("=" * 65)


# ------------------------------------------------------------------
# Random instance generator
# ------------------------------------------------------------------

def random_instance(n: int, capacity: float = 10, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [float(rng.randint(1, int(capacity) // 2)) for _ in range(n)]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Original instance ----
    demands_orig = [4.0, 3.0, 5.0, 2.0, 4.0, 3.0]
    cap = 10.0
    n_orig = len(demands_orig)

    lb = lb_fractional(demands_orig, cap)
    n_ffd, bins_ffd = first_fit_decreasing(demands_orig, cap)

    print("=" * 65)
    print("PROBLEMA 4 - Instancia do enunciado (6 VMs, cap=10)")
    print(f"Demandas: {demands_orig}")
    print(f"Limitante inferior (fracionario): {lb} servidores")
    print(f"Limitante superior (FFD): {n_ffd} servidores")
    for j, vms in enumerate(bins_ffd):
        load = sum(demands_orig[i] for i in vms)
        print(f"  FFD Bin {j+1}: {['VM'+str(i+1) for i in vms]} -> carga={load:.0f}")
    print("=" * 65)

    def _run_instance(demands, cap, n_ffd, bins_ffd, label, tl=30.0):
        """Solve one instance with B&B and B&C, using FFD as warm-start incumbent."""
        k = n_ffd
        lb = lb_fractional(demands, cap)
        model = build_model(demands, cap, max_servers=k)

        # Warm-start: encode FFD solution as initial incumbent
        x_ffd = ffd_to_x_vector(bins_ffd, len(demands), k)

        # If LP lower bound equals FFD upper bound, optimality is already proved.
        if lb == n_ffd:
            print(f"  LB={lb} == FFD={n_ffd}: otimalidade provada pelo limitante, B&B confirma.")

        print("\n--- Branch-and-Bound ---")
        bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible",
                            time_limit=tl,
                            initial_incumbent=float(n_ffd), initial_x=x_ffd)
        r_bb = bb.solve()
        print_solution(r_bb, demands, cap, f"B&B | {label}", max_servers=k)

        print("\n--- Branch-and-Cut (cover cuts) ---")
        bc = BranchAndCut(model, strategy="best_first", branching="most_infeasible",
                          cut_types=["cover"], time_limit=tl,
                          initial_incumbent=float(n_ffd), initial_x=x_ffd)
        r_bc = bc.solve()
        print_solution(r_bc, demands, cap, f"B&C | {label}", max_servers=k)
        print("\nLog de cortes B&C:")
        print_cut_log(bc, label=label)

        return r_bb, r_bc

    # Original instance
    model_orig = build_model(demands_orig, cap, max_servers=n_ffd)
    r_bb, r_bc = _run_instance(demands_orig, cap, n_ffd, bins_ffd, "Instancia original")
    summary = [("Original (6 VMs)", r_bb, r_bc, n_ffd)]

    # ---- Random instances ----
    for n_vms, seed in [(10, 1), (15, 2), (20, 3)]:
        demands = random_instance(n_vms, cap, seed=seed)
        lb_r = lb_fractional(demands, cap)
        n_ffd_r, bins_ffd_r = first_fit_decreasing(demands, cap)
        label = f"Aleatoria ({n_vms} VMs)"

        print(f"\n{'=' * 65}")
        print(f"PROBLEMA 4 - {label}, cap={cap:.0f}")
        print(f"Demandas: {demands}")
        print(f"LB={lb_r} | FFD={n_ffd_r} (pool={n_ffd_r} servidores)")
        print("=" * 65)

        r_bb_r, r_bc_r = _run_instance(demands, cap, n_ffd_r, bins_ffd_r, label, tl=30.0)
        summary.append((label, r_bb_r, r_bc_r, n_ffd_r))

    # ---- Summary ----
    print("\n--- Resumo ---")
    hdr = f"{'Instancia':<24} {'FFD':>5} {'B&B':>5} {'B&B nos':>9} {'B&C':>5} {'B&C nos':>9} {'Reducao':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, rbb, rbc, ffd in summary:
        bb_obj = f"{round(rbb.obj_value)}" if rbb.obj_value is not None else "?"
        bc_obj = f"{round(rbc.obj_value)}" if rbc.obj_value is not None else "?"
        if rbb.nodes_explored > 0 and rbc.nodes_explored > 0:
            red = f"{100*(1 - rbc.nodes_explored/rbb.nodes_explored):.1f}%"
        else:
            red = "N/A"
        print(f"{label:<24} {ffd:>5} {bb_obj:>5} {rbb.nodes_explored:>9} "
              f"{bc_obj:>5} {rbc.nodes_explored:>9} {red:>9}")
