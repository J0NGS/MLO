"""
Problema 3 - Seleção de Fluxos em Rede de Datacenter [PIB]

Knapsack Multidimensional: selecionar fluxos maximizando prioridade
total sujeita a duas restrições de capacidade (banda e buffer).

Formulação:
    max  sum_j priority[j] * x[j]
    s.t.
    (1) sum_j banda[j]  * x[j] <= banda_cap    (banda máxima)
    (2) sum_j buffer[j] * x[j] <= buffer_cap   (buffer máximo)
        x[j] in {0,1}
"""
from __future__ import annotations
import random
import time
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult
from core.relaxation import solve_relaxation


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    banda: list[float],
    buffer: list[float],
    priority: list[float],
    banda_cap: float,
    buffer_cap: float,
) -> MIPModel:
    n = len(banda)
    assert len(buffer) == n and len(priority) == n

    A_ub = [banda, buffer]
    b_ub = [float(banda_cap), float(buffer_cap)]

    return MIPModel(
        c=list(priority),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0.0, 1.0)] * n,
        integrality=[2] * n,
        sense="max",
        var_names=[f"F{j+1}" for j in range(n)],
    )


# ------------------------------------------------------------------
# Fractional knapsack bound (greedy by priority/banda ratio)
# Used as a quick upper-bound heuristic for comparison.
# The LP relaxation used by B&B is tighter (considers both constraints).
# ------------------------------------------------------------------

def fractional_knapsack_bound(
    banda: list[float],
    buffer: list[float],
    priority: list[float],
    banda_cap: float,
    buffer_cap: float,
) -> float:
    """
    Greedy fractional knapsack upper bound using priority/banda ratio.
    Ignores the buffer constraint, so it is an optimistic (loose) upper bound.
    """
    items = sorted(
        range(len(banda)),
        key=lambda j: priority[j] / banda[j],
        reverse=True,
    )
    cap = float(banda_cap)
    bound = 0.0
    for j in items:
        if cap <= 0:
            break
        frac = min(1.0, cap / banda[j])
        bound += frac * priority[j]
        cap -= frac * banda[j]
    return bound


def greedy_integer_solution(
    banda: list[float],
    buffer: list[float],
    priority: list[float],
    banda_cap: float,
    buffer_cap: float,
) -> tuple[float, np.ndarray]:
    """
    Greedy integer solution using priority/banda ratio.
    Respects both bandwidth and buffer constraints (no fractional items).
    Returns (total_priority, x_vector) — a feasible incumbent for B&B.
    """
    items = sorted(
        range(len(banda)),
        key=lambda j: priority[j] / banda[j],
        reverse=True,
    )
    rem_banda  = float(banda_cap)
    rem_buffer = float(buffer_cap)
    x = np.zeros(len(banda))
    total_priority = 0.0
    for j in items:
        if banda[j] <= rem_banda + 1e-9 and buffer[j] <= rem_buffer + 1e-9:
            x[j] = 1.0
            rem_banda  -= banda[j]
            rem_buffer -= buffer[j]
            total_priority += priority[j]
    return total_priority, x


# ------------------------------------------------------------------
# Solution printer
# ------------------------------------------------------------------

def print_solution(
    result: BBResult,
    banda: list[float],
    buffer: list[float],
    priority: list[float],
    banda_cap: float,
    buffer_cap: float,
    label: str = "",
):
    n = len(banda)
    print("\n" + "=" * 65)
    if label:
        print(f"  {label}")
    print(f"Status         : {result.status}")
    print(f"Nos explorados : {result.nodes_explored}")
    print(f"Tempo (s)      : {result.elapsed:.4f}")

    if result.x is None:
        print("Sem solucao viavel.")
        print("=" * 65)
        return

    print(f"Prioridade opt.: {result.obj_value:.2f}")

    selected = [j for j in range(n) if round(result.x[j]) == 1]
    total_banda = sum(banda[j] for j in selected)
    total_buffer = sum(buffer[j] for j in selected)

    print(f"Fluxos selecionados: {[f'F{j+1}' for j in selected]}")
    print(f"Banda usada   : {total_banda:.1f} / {banda_cap} Mbps")
    print(f"Buffer usado  : {total_buffer:.1f} / {buffer_cap} MB")
    print("=" * 65)


# ------------------------------------------------------------------
# Random instance generator
# ------------------------------------------------------------------

def random_instance(n: int, seed: int = 0):
    rng = random.Random(seed)
    banda   = [rng.randint(10, 50) for _ in range(n)]
    buffer  = [rng.randint(2, 15)  for _ in range(n)]
    priority = [rng.randint(10, 100) for _ in range(n)]
    return banda, buffer, priority


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Instance from the assignment ----
    banda_orig    = [30, 20, 45, 15, 35, 25, 40]
    buffer_orig   = [8,  5,  10,  3,  9,  6,  7]
    priority_orig = [50, 30, 70, 20, 60, 40, 65]
    banda_cap  = 100
    buffer_cap = 25

    fk_bound = fractional_knapsack_bound(
        banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap
    )
    greedy_val, x_greedy = greedy_integer_solution(
        banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap
    )

    print("=" * 65)
    print("PROBLEMA 3 - Instancia do enunciado (7 fluxos)")
    print(f"Banda cap={banda_cap} Mbps | Buffer cap={buffer_cap} MB")
    print(f"Limitante superior fracionario (greedy razao p/b): {fk_bound:.2f}")
    print(f"Solucao inicial greedy inteira (incumbente B&B):  {greedy_val:.2f}")
    print("=" * 65)

    model_orig = build_model(banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap)

    lp_orig = solve_relaxation(model_orig)
    lp_bound_orig = lp_orig.obj_value if lp_orig.status == "optimal" else None

    print("\n--- Branch-and-Bound (warm-start: solucao greedy inteira) ---")
    bb = BranchAndBound(model_orig, strategy="best_first", branching="most_infeasible",
                        initial_incumbent=greedy_val, initial_x=x_greedy)
    r_bb = bb.solve()
    print_solution(r_bb, banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap,
                   "B&B | Instancia original")

    print("\n--- Branch-and-Cut (cover cuts, warm-start: greedy inteira) ---")
    bc = BranchAndCut(model_orig, strategy="best_first", branching="most_infeasible",
                      cut_types=["cover"],
                      initial_incumbent=greedy_val, initial_x=x_greedy)
    r_bc = bc.solve()
    print_solution(r_bc, banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap,
                   "B&C | Instancia original")

    # ---- Random instances ----
    summary_rows = [
        ("Original (7 fluxos)", r_bb, r_bc, lp_bound_orig),
    ]

    for n_flows, seed in [(10, 1), (20, 2), (50, 3)]:
        banda, buffer, priority = random_instance(n_flows, seed=seed)
        # Scale capacities proportionally to instance size
        scale = n_flows / 7
        bc_cap  = int(banda_cap  * scale * 0.6)
        buf_cap = int(buffer_cap * scale * 0.6)

        label = f"Aleatoria ({n_flows} fluxos)"
        print(f"\n{'=' * 65}")
        print(f"PROBLEMA 3 - {label}")
        print(f"Banda cap={bc_cap} Mbps | Buffer cap={buf_cap} MB")
        print("=" * 65)

        model_r = build_model(banda, buffer, priority, bc_cap, buf_cap)
        lp_r = solve_relaxation(model_r)
        lp_bound_r = lp_r.obj_value if lp_r.status == "optimal" else None

        g_val_r, x_g_r = greedy_integer_solution(banda, buffer, priority, bc_cap, buf_cap)

        print("\n--- Branch-and-Bound ---")
        bb_r = BranchAndBound(model_r, strategy="best_first", branching="most_infeasible",
                               initial_incumbent=g_val_r if g_val_r > 0 else None,
                               initial_x=x_g_r if g_val_r > 0 else None)
        r_bb_r = bb_r.solve()
        print_solution(r_bb_r, banda, buffer, priority, bc_cap, buf_cap,
                       f"B&B | {label}")

        print("\n--- Branch-and-Cut (cover cuts) ---")
        bc_r = BranchAndCut(model_r, strategy="best_first", branching="most_infeasible",
                             cut_types=["cover"],
                             initial_incumbent=g_val_r if g_val_r > 0 else None,
                             initial_x=x_g_r if g_val_r > 0 else None)
        r_bc_r = bc_r.solve()
        print_solution(r_bc_r, banda, buffer, priority, bc_cap, buf_cap,
                       f"B&C | {label}")

        summary_rows.append((label, r_bb_r, r_bc_r, lp_bound_r))

    # ---- Summary ----
    print("\n--- Resumo das instancias ---")
    print(f"{'Instancia':<25} {'LP bound':>10} {'B&B obj':>10} {'Gap%':>7} {'B&B nos':>9} {'B&C obj':>10} {'B&C nos':>9} {'Reducao':>9}")
    print("-" * 100)
    for label, rbb, rbc, lp_bound in summary_rows:
        obj_bb  = f"{rbb.obj_value:.1f}" if rbb.obj_value is not None else "N/A"
        obj_bc  = f"{rbc.obj_value:.1f}" if rbc.obj_value is not None else "N/A"
        lp_str  = f"{lp_bound:.2f}" if lp_bound is not None else "N/A"
        if lp_bound is not None and rbb.obj_value is not None and abs(lp_bound) > 1e-8:
            gap = abs(lp_bound - rbb.obj_value) / abs(lp_bound) * 100
            gap_str = f"{gap:.1f}%"
        else:
            gap_str = "N/A"
        if rbb.nodes_explored > 0:
            reducao = f"{100*(1 - rbc.nodes_explored/rbb.nodes_explored):.1f}%"
        else:
            reducao = "N/A"
        print(f"{label:<25} {lp_str:>10} {obj_bb:>10} {gap_str:>7} {rbb.nodes_explored:>9} {obj_bc:>10} {rbc.nodes_explored:>9} {reducao:>9}")
