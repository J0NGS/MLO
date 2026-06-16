"""
Problema 9 - Empacotamento de Caixas (Bin Packing) [PIB]

Uma transportadora precisa empacotar n itens em caixas identicas
de capacidade B, minimizando o numero de caixas utilizadas.

Formulacao PIB:
    min  sum_j y_j
    s.t.
    (1)  sum_j x_{ij} = 1               para cada item i      (atribuicao unica)
    (2)  sum_i w_i * x_{ij} <= B * y_j  para cada caixa j     (capacidade)
    (3)  y_j <= y_{j-1}                 para j = 1..n-1       (quebra de simetria)
    (4)  y_j, x_{ij} in {0, 1}                                (PIB puro)

Variaveis (n_bins + n_items*n_bins variaveis):
    y_j    -> j                         (j = 0..n_bins-1)
    x_{ij} -> n_bins + i*n_bins + j     (i = 0..n_items-1)

Instancia (6 itens, capacidade B=10):
    Pesos: [6, 4, 4, 3, 3, 2]   soma = 22
    Limite inferior: ceil(22/10) = 3 caixas

    Otimo: 3 caixas
        B1: {I1(6), I2(4)}       = 10
        B2: {I3(4), I4(3), I5(3)}= 10
        B3: {I6(2)}              =  2

Heuristica FFD (First-Fit Decreasing):
    Ordena decrescente: [6,4,4,3,3,2]
    B1: 6 -> 6+2=8 (cabe 2); B2: 4+4=8; B3: 3+3=6.  -> 3 caixas (=otimo)
    NOTA: para B=11, FFD da 3 mas otimo e 2 (FFD e subotimo!).

Analise de sensibilidade (capacidade B em [6..13]):
    B =  6, 7  -> 4 caixas  (3*7=21 < 22 = soma)
    B =  8..10 -> 3 caixas
    B = 11..13 -> 2 caixas  (B=11: {6,3,2},{4,4,3}; mas FFD da 3 - subotimo!)
"""
from __future__ import annotations
import math
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult


# ------------------------------------------------------------------
# Default instance
# ------------------------------------------------------------------

N_ITEMS = 6
N_BINS  = 6   # at most n_items bins ever needed

ITEM_NAMES = [f"I{i+1}" for i in range(N_ITEMS)]
BIN_NAMES  = [f"B{j+1}" for j in range(N_BINS)]

DEFAULT_WEIGHTS:  list[float] = [6.0, 4.0, 4.0, 3.0, 3.0, 2.0]
DEFAULT_CAPACITY: float       = 10.0


# ------------------------------------------------------------------
# Variable index helpers
# ------------------------------------------------------------------

def iy(j: int, n_bins: int) -> int:
    """Index of binary variable y_j (bin j is open)."""
    return j

def ix(i: int, j: int, n_bins: int) -> int:
    """Index of binary variable x_{ij} (item i in bin j)."""
    return n_bins + i * n_bins + j


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    weights: list[float],
    capacity: float,
    n_bins: int | None = None,
    item_names: list[str] | None = None,
    bin_names: list[str] | None = None,
) -> MIPModel:
    """
    Build Bin Packing MIPModel with symmetry-breaking constraints.

    All variables are binary (PIB puro).
    """
    n_items = len(weights)
    if n_bins is None:
        n_bins = n_items   # worst case: one item per bin
    if item_names is None:
        item_names = [f"i{i+1}" for i in range(n_items)]
    if bin_names is None:
        bin_names = [f"b{j+1}" for j in range(n_bins)]

    n_vars = n_bins + n_items * n_bins

    # Objective: minimize number of open bins
    c = [0.0] * n_vars
    for j in range(n_bins):
        c[iy(j, n_bins)] = 1.0

    bounds = [(0.0, 1.0)] * n_vars
    integrality = [2] * n_vars   # all binary

    A_ub, b_ub = [], []

    # (2) Capacity: sum_i w_i x_{ij} - B y_j <= 0
    for j in range(n_bins):
        row = [0.0] * n_vars
        for i in range(n_items):
            row[ix(i, j, n_bins)] = float(weights[i])
        row[iy(j, n_bins)] = -float(capacity)
        A_ub.append(row)
        b_ub.append(0.0)

    # (3) Symmetry breaking: y_j - y_{j-1} <= 0
    for j in range(1, n_bins):
        row = [0.0] * n_vars
        row[iy(j - 1, n_bins)] = -1.0
        row[iy(j, n_bins)]     =  1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # (1) Assignment: sum_j x_{ij} = 1
    A_eq, b_eq = [], []
    for i in range(n_items):
        row = [0.0] * n_vars
        for j in range(n_bins):
            row[ix(i, j, n_bins)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    var_names  = list(bin_names)
    var_names += [f"x{item_names[i]}{bin_names[j]}"
                  for i in range(n_items) for j in range(n_bins)]

    return MIPModel(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, integrality=integrality,
        sense="min",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# First-Fit Decreasing heuristic
# ------------------------------------------------------------------

def ffd_assignment(
    weights: list[float],
    capacity: float,
) -> tuple[dict[int, int], int]:
    """
    First-Fit Decreasing heuristic.
    Returns (item_bin_map, n_bins_used) where item_bin_map[i] = j.
    """
    order = sorted(range(len(weights)), key=lambda i: -weights[i])
    bin_loads: list[float] = []
    item_bin: dict[int, int] = {}

    for i in order:
        placed = False
        for j, load in enumerate(bin_loads):
            if load + weights[i] <= capacity + 1e-9:
                bin_loads[j] += weights[i]
                item_bin[i] = j
                placed = True
                break
        if not placed:
            item_bin[i] = len(bin_loads)
            bin_loads.append(float(weights[i]))

    return item_bin, len(bin_loads)


def ffd_to_x_vector(
    item_bin: dict[int, int],
    n_items: int,
    n_bins: int,
) -> np.ndarray:
    x = np.zeros(n_bins + n_items * n_bins)
    for i, j in item_bin.items():
        if j < n_bins:
            x[iy(j, n_bins)]      = 1.0
            x[ix(i, j, n_bins)]   = 1.0
    return x


# ------------------------------------------------------------------
# Solution decoder and printer
# ------------------------------------------------------------------

def decode_solution(
    result: BBResult,
    n_items: int,
    n_bins: int,
) -> tuple[int, dict[int, list[int]]]:
    """
    Returns (n_open_bins, bin_contents) where bin_contents[j] = [item indices].
    """
    if result.x is None:
        return 0, {}
    open_bins = [j for j in range(n_bins) if result.x[iy(j, n_bins)] > 0.5]
    bin_contents = {
        j: [i for i in range(n_items) if result.x[ix(i, j, n_bins)] > 0.5]
        for j in open_bins
    }
    return len(open_bins), bin_contents


def print_solution(
    result: BBResult,
    weights: list[float],
    capacity: float,
    label: str = "",
    item_names: list[str] | None = None,
    bin_names: list[str] | None = None,
):
    n_items = len(weights)
    n_bins  = n_items
    if item_names is None:
        item_names = [f"I{i+1}" for i in range(n_items)]
    if bin_names is None:
        bin_names  = [f"B{j+1}" for j in range(n_bins)]

    print("\n" + "=" * 62)
    if label:
        print(f"  {label}")
    print(f"Status          : {result.status}")
    print(f"Nos explorados  : {result.nodes_explored}")
    print(f"Tempo (s)       : {result.elapsed:.4f}")

    if result.x is None:
        print("Sem solucao viavel.")
        print("=" * 62)
        return

    n_open, bin_contents = decode_solution(result, n_items, n_bins)
    print(f"Caixas usadas   : {result.obj_value:.0f}")
    print()
    print(f"  {'Caixa':<6} {'Itens':<35}  {'Carga':>6}")
    print("  " + "-" * 50)
    for j, items in sorted(bin_contents.items()):
        load = sum(weights[i] for i in items)
        items_str = ", ".join(
            f"{item_names[i]}({weights[i]:.0f})" for i in sorted(items)
        )
        print(f"  {bin_names[j]:<6} {items_str:<35}  {load:.0f}/{capacity:.0f}")
    print("=" * 62)


# ------------------------------------------------------------------
# Sensitivity analysis: vary bin capacity
# ------------------------------------------------------------------

def sensitivity_analysis(
    cap_range: list[float],
    weights: list[float] = DEFAULT_WEIGHTS,
) -> list[tuple[float, float | None, int]]:
    """
    Vary capacity B over cap_range.
    Returns list of (B, optimal_bins, ffd_bins).
    """
    n_items = len(weights)
    n_bins  = n_items
    results = []
    for B in cap_range:
        model = build_model(weights, B, n_bins)
        item_bin, ffd_n = ffd_assignment(weights, B)
        g_x = ffd_to_x_vector(item_bin, n_items, n_bins)
        r = BranchAndBound(
            model, strategy="best_first", branching="first_fractional",
            initial_incumbent=float(ffd_n), initial_x=g_x,
            verbose=False,
        ).solve()
        results.append((B, r.obj_value, ffd_n))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    fn = ITEM_NAMES
    bn = BIN_NAMES
    W  = DEFAULT_WEIGHTS
    B  = DEFAULT_CAPACITY

    print("=" * 62)
    print("PROBLEMA 9 - Empacotamento de Caixas (Bin Packing PIB)")
    print("=" * 62)

    total_w = sum(W)
    lb = math.ceil(total_w / B)
    print(f"\n  {N_ITEMS} itens | capacidade B = {B:.0f}")
    print(f"  Pesos : {W}")
    print(f"  Soma  : {total_w:.0f}")
    print(f"  Limite inferior LP: ceil({total_w:.0f}/{B:.0f}) = {lb}")

    model = build_model(W, B, N_BINS, fn, bn)

    item_bin, ffd_n = ffd_assignment(W, B)
    g_x = ffd_to_x_vector(item_bin, N_ITEMS, N_BINS)

    print(f"\n  Heuristica FFD: {ffd_n} caixas")
    for j in sorted(set(item_bin.values())):
        items_in = sorted(i for i, b in item_bin.items() if b == j)
        load = sum(W[i] for i in items_in)
        s = ", ".join(f"{fn[i]}({W[i]:.0f})" for i in items_in)
        print(f"    {bn[j]}: [{s}] = {load:.0f}")

    print("\n--- Branch-and-Bound ---")
    bb = BranchAndBound(
        model, strategy="best_first", branching="first_fractional",
        initial_incumbent=float(ffd_n), initial_x=g_x,
    )
    r_bb = bb.solve()
    print_solution(r_bb, W, B, label="B&B | Instancia original",
                   item_names=fn, bin_names=bn)

    print("\n--- Branch-and-Cut (cortes de Gomory) ---")
    bc = BranchAndCut(
        model, strategy="best_first", branching="first_fractional",
        cut_types=["gomory"],
        initial_incumbent=float(ffd_n), initial_x=g_x,
    )
    r_bc = bc.solve()
    print_solution(r_bc, W, B, label="B&C | Instancia original",
                   item_names=fn, bin_names=bn)

    print(f"\nB&B nos={r_bb.nodes_explored} | B&C nos={r_bc.nodes_explored}")

    # ---- Sensitivity: vary B ----
    print("\n" + "=" * 62)
    print("ANALISE DE SENSIBILIDADE — capacidade B")
    print("=" * 62)
    cap_range = [float(v) for v in range(6, 14)]
    sens = sensitivity_analysis(cap_range, W)

    print(f"\n  {'B':>5}  {'LB=ceil(22/B)':>14}  {'Otimo':>7}  {'FFD':>5}  {'FFD=Otimo?':>10}")
    print("  " + "-" * 52)
    prev_opt = None
    for B_val, n_opt, ffd_bins in sens:
        lb_val = math.ceil(total_w / B_val)
        marker = " <-- muda" if n_opt != prev_opt and prev_opt is not None else ""
        ffd_ok = "SIM" if ffd_bins == n_opt else "NAO *"
        print(f"  {B_val:>5.0f}  {lb_val:>14}  {n_opt:>7.0f}  {ffd_bins:>5}  {ffd_ok:>10}{marker}")
        prev_opt = n_opt

    print("\n  * FFD subotimo: B=11, FFD da 3 caixas mas otimo e 2")
    print("\nInterpretacao:")
    print(f"  B =  6, 7  -> 4 caixas (ceil(22/B) >= 4, 3 caixas nao cabem)")
    print(f"  B =  8-10  -> 3 caixas  ex.: {{I1,I2}}, {{I3,I4,I5}}, {{I6}}")
    print(f"  B = 11-13  -> 2 caixas  ex.: {{I1,I4,I6}}, {{I2,I3,I5}}")
    print(f"  Transicoes: B=8 (4->3 caixas), B=11 (3->2 caixas)")
