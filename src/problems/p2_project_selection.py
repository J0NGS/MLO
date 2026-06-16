"""
Problema 2 - Seleção de Projetos de Pesquisa com Restrições Lógicas [PIB]

Maximize sum_i r_i(ponto de impacto) * x_i(variavel de seleção binária)
Subject to:
    sum_i c_i * x_i <= 280          (orçamento, onde c_i é o custo do projeto i)
    x3 <= x1                         (P3 só se P1) → x3 - x1 <= 0
    x4 + x5 <= 1                     (P4 e P5 mutuamente exclusivos)
    x1 + x2 + x4 >= 2               (ao menos 2 de {P1,P2,P4}) → -x1 - x2 - x4 <= -2
    xi in {0, 1}   for all i

Variables: x0..x5 = P1..P6 (0-indexed)
"""
from __future__ import annotations
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut


def build_model() -> MIPModel:
    # traduz o problema real em vetores e matrizes pra passar pro MIPModel
    costs  = [80,  60,  90,  50,  70,  100]   # custo de cada projeto p1..p6 (em R$ mil)
    impact = [120, 85, 140,  60, 110,  160]   # impacto (pontos) de cada projeto
    budget = 280
    n = 6

    # função objetivo; maximizar impacto total; effective_c() inverte o sinal internamente pro linprog
    c = list(impact)

    # cada append adiciona uma linha nova à matriz A_ub; formato: A_ub @ x <= b_ub
    A_ub = []
    b_ub = []

    # restrição 1; orçamento; soma dos custos dos projetos selecionados não pode passar de 280
    A_ub.append(costs)
    b_ub.append(budget)

    # restrição 2; dependência lógica; p3 só pode ser feito se p1 também for; x[2] - x[0] <= 0
    row = [0] * n
    row[2] = 1
    row[0] = -1
    A_ub.append(row)
    b_ub.append(0)

    # restrição 3; exclusão mútua; p4 e p5 não podem ser feitos ao mesmo tempo; x[3] + x[4] <= 1
    row = [0] * n
    row[3] = 1
    row[4] = 1
    A_ub.append(row)
    b_ub.append(1)

    # restrição 4; mínimo de 2 de {p1, p2, p4}; x[0]+x[1]+x[3] >= 2 vira -x[0]-x[1]-x[3] <= -2
    # o sinal é invertido pra caber no formato A_ub @ x <= b_ub que o modelo exige
    row = [0] * n
    row[0] = -1
    row[1] = -1
    row[3] = -1
    A_ub.append(row)
    b_ub.append(-2)

    # monta o modelo; todas as variáveis são binárias (0 ou 1); problema de maximização
    return MIPModel(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, 1)] * n,
        integrality=[2] * n,   # all binary
        sense="max",
        var_names=[f"P{i+1}" for i in range(n)],
    )


def print_solution(result, model: MIPModel):
    print("\n" + "=" * 60)
    print(f"Status        : {result.status}")
    print(f"Nós explorados: {result.nodes_explored}")
    print(f"Tempo (s)     : {result.elapsed:.4f}")
    if result.x is not None:
        print(f"Impacto total : {result.obj_value:.0f} pts")
        custo = sum(ci * round(xi) for ci, xi in zip([80, 60, 90, 50, 70, 100], result.x))
        print(f"Custo total   : {custo:.0f} R$ mil")
        print("\nProjetos selecionados:")
        for i, xi in enumerate(result.x):
            selected = round(xi) == 1
            print(f"  {model.var_names[i]}: {'APROVADO' if selected else 'rejeitado'}  (x={xi:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    # constrói o modelo uma vez; os dois solvers usam o mesmo objeto
    model = build_model()

    print("\n" + "=" * 60)
    print("PROBLEMA 2 — Branch-and-Bound (best-first, most-infeasible)")
    print("=" * 60)
    # b&b puro; best_first sempre processa o nó com melhor bound lp; most_infeasible ramifica na variável mais próxima de 0.5
    solver_bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible")
    result_bb = solver_bb.solve()
    print_solution(result_bb, model)

    print("\n" + "=" * 60)
    print("PROBLEMA 2 — Branch-and-Cut (Gomory cuts)")
    print("=" * 60)
    # b&c; mesma estratégia de exploração; mas tenta adicionar cortes de gomory antes de ramificar
    # cortes apertam o lp; tendem a reduzir o número de nós explorados
    solver_bc = BranchAndCut(model, strategy="best_first", branching="most_infeasible",
                              cut_types=["gomory"])
    result_bc = solver_bc.solve()
    print_solution(result_bc, model)

    # compara os dois; menos nós = b&c foi mais eficiente; mais nós = cortes não compensaram aqui
    print("\n--- Comparação B&B vs B&C ---")
    print(f"B&B nós: {result_bb.nodes_explored}  |  B&C nós: {result_bc.nodes_explored}")
    print(f"B&B tempo: {result_bb.elapsed:.4f}s  |  B&C tempo: {result_bc.elapsed:.4f}s")
