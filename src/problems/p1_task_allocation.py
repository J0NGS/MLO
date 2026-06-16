"""
Problema 1 - Alocação de Tarefas em Sistema Multiprocessador [PLIP]

Minimize M  (makespan = tempo máximo entre servidores)

Variáveis:
    x[i][j] ∈ {0,1} : tarefa j alocada ao servidor i  (i=0..S-1, j=0..T-1)
    M >= 0 (contínua): makespan auxiliar

Formulação:
    Minimize M
    s.t.
    (1) sum_j p[i][j] * x[i][j] <= M       ∀ i  (linearização do máximo)
    (2) sum_j p[i][j] * x[i][j] <= cap     ∀ i  (capacidade do servidor)
    (3) sum_i x[i][j]  = 1                 ∀ j  (cada tarefa em exatamente 1 servidor)
        x[i][j] ∈ {0,1},   M >= 0

Índice das variáveis (flattened):
    x[i][j]  →  i * n_tasks + j   (primeiros S*T variáveis)
    M        →  índice S*T
"""
from __future__ import annotations
import time
import random
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BBResult


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    p: list[list[float]],
    capacity: float,
) -> MIPModel:
    """
    Build the MIPModel for the task-allocation problem.

    Parameters
    ----------
    p        : p[i][j] = execution time of task j on server i
    capacity : maximum load per server
    """
    S = len(p)          # number of servers
    T = len(p[0])       # number of tasks
    n_vars = S * T + 1  # binary + M
    idx_M = S * T       # index of M in the variable vector

    # Objective: minimize M
    c = [0.0] * n_vars
    c[idx_M] = 1.0

    # Variable bounds: x_ij ∈ [0,1], M ∈ [0, ∞)
    bounds = [(0.0, 1.0)] * (S * T) + [(0.0, None)]

    # Integrality: 2=binary for x_ij, 0=continuous for M
    integrality = [2] * (S * T) + [0]

    # ---- Inequality constraints (A_ub @ x <= b_ub) ----
    A_ub, b_ub = [], []

    for i in range(S):
        # (1) Makespan: sum_j p[i][j]*x[i][j] - M <= 0
        row = [0.0] * n_vars
        for j in range(T):
            row[i * T + j] = p[i][j]
        row[idx_M] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    for i in range(S):
        # (2) Capacity: sum_j p[i][j]*x[i][j] <= capacity
        row = [0.0] * n_vars
        for j in range(T):
            row[i * T + j] = p[i][j]
        A_ub.append(row)
        b_ub.append(float(capacity))

    # ---- Equality constraints (A_eq @ x = b_eq) ----
    A_eq, b_eq = [], []

    for j in range(T):
        # (3) Each task assigned to exactly one server
        row = [0.0] * n_vars
        for i in range(S):
            row[i * T + j] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    # Variable names
    var_names = [f"x{i+1}{j+1}" for i in range(S) for j in range(T)] + ["M"]

    return MIPModel(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=integrality,
        sense="min",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# Solution printer
# ------------------------------------------------------------------

def print_solution(result: BBResult, p: list[list[float]], capacity: float):
    S, T = len(p), len(p[0])
    print("\n" + "=" * 65)
    print(f"Status         : {result.status}")
    print(f"Nós explorados : {result.nodes_explored}")
    print(f"Tempo (s)      : {result.elapsed:.4f}")

    if result.x is None:
        print("Sem solução viável.")
        return

    print(f"Makespan ótimo : {result.obj_value:.2f} s")

    # Decode assignment
    assignment: dict[int, list[int]] = {i: [] for i in range(S)}
    for i in range(S):
        for j in range(T):
            if round(result.x[i * T + j]) == 1:
                assignment[i].append(j)

    print("\nAlocação de tarefas:")
    for i in range(S):
        tasks = assignment[i]
        load = sum(p[i][j] for j in tasks)
        task_str = ", ".join(f"T{j+1}({p[i][j]}s)" for j in tasks) if tasks else "—"
        print(f"  S{i+1}: {task_str}  ->  carga = {load}s")

    print("=" * 65)


# ------------------------------------------------------------------
# Random instance generator
# ------------------------------------------------------------------

def random_instance(
    n_servers: int,
    n_tasks: int,
    capacity: int,
    seed: int = 0,
) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [rng.randint(1, capacity // 2) for _ in range(n_tasks)]
        for _ in range(n_servers)
    ]


# ------------------------------------------------------------------
# Main: run original + 2 random instances
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Instance 1: original from assignment ----
    p_orig = [
        [4, 5, 3, 7],   # S1
        [3, 6, 4, 5],   # S2
        [5, 4, 6, 3],   # S3
    ]
    cap_orig = 12

    print("=" * 65)
    print("PROBLEMA 1 — Instância do enunciado (3 servidores, 4 tarefas)")
    print("=" * 65)
    model_orig = build_model(p_orig, cap_orig)
    solver_orig = BranchAndBound(model_orig, strategy="best_first",
                                 branching="most_infeasible")
    result_orig = solver_orig.solve()
    print_solution(result_orig, p_orig, cap_orig)

    # ---- Instance 2: random, 5 servers, 8 tasks ----
    p2 = random_instance(5, 8, 20, seed=1)
    cap2 = 20

    print("\n" + "=" * 65)
    print("PROBLEMA 1 — Instância aleatória 2 (5 servidores, 8 tarefas)")
    print("=" * 65)
    print("Matriz de tempos (servidores × tarefas):")
    for i, row in enumerate(p2):
        print(f"  S{i+1}: {row}")
    model2 = build_model(p2, cap2)
    solver2 = BranchAndBound(model2, strategy="best_first",
                              branching="most_infeasible")
    result2 = solver2.solve()
    print_solution(result2, p2, cap2)

    # ---- Instance 3: random, 6 servers, 10 tasks ----
    p3 = random_instance(6, 10, 25, seed=42)
    cap3 = 25

    print("\n" + "=" * 65)
    print("PROBLEMA 1 — Instância aleatória 3 (6 servidores, 10 tarefas)")
    print("=" * 65)
    print("Matriz de tempos (servidores × tarefas):")
    for i, row in enumerate(p3):
        print(f"  S{i+1}: {row}")
    model3 = build_model(p3, cap3)
    solver3 = BranchAndBound(model3, strategy="best_first",
                              branching="most_infeasible")
    result3 = solver3.solve()
    print_solution(result3, p3, cap3)

    # ---- Summary table ----
    print("\n--- Resumo das instâncias ---")
    print(f"{'Instância':<35} {'Makespan':>10} {'Nós':>8} {'Tempo (s)':>12}")
    print("-" * 70)
    for label, res in [
        ("Original (3s × 4t, cap=12)", result_orig),
        ("Aleatória 2 (5s × 8t, cap=20)", result2),
        ("Aleatória 3 (6s × 10t, cap=25)", result3),
    ]:
        obj = f"{res.obj_value:.2f}" if res.obj_value is not None else "N/A"
        print(f"{label:<35} {obj:>10} {res.nodes_explored:>8} {res.elapsed:>12.4f}")
