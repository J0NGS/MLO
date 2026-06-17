"""Geradores de cortes para Branch-and-Cut.

Cada gerador retorna uma lista de (lhs_row, rhs_scalar) que satisfaz lhs @ x <= rhs.
Os cortes devem ser GLOBALMENTE válidos: não podem remover nenhum ponto inteiro factível.
"""
from __future__ import annotations
import math
import itertools
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MIPModel


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _frac(v: float) -> float:
    # Parte fracionária no intervalo [0, 1); usada nas fórmulas de Gomory.
    f = v - math.floor(v)
    return f


def _find_basis(A_full, b_bar, x_aug, m, tol=1e-5):
    """
    Encontra uma base m×m invertível para o LP na solução ótima atual.

    Busca combinações de tamanho m a partir de (variáveis fracionárias) +
    (candidatas em zero para básicas degeneradas) até encontrar uma B invertível com
    B @ x_B ≈ b_bar.

    Retorna (basic_idx, B_inv, x_B) ou (None, None, None) em caso de falha.
    """
    n_aug = A_full.shape[1]

    # Variáveis estritamente fracionárias tendem a aparecer como básicas
    # no ótimo do LP; priorizamos essas colunas na busca.
    definite = [
        i for i in range(n_aug)
        if x_aug[i] > tol and not math.isclose(x_aug[i], round(x_aug[i]), abs_tol=tol)
    ]

    # Variáveis positivas com valor inteiro podem ou não ser básicas;
    # entram como candidatas secundárias.
    positive_int = [
        i for i in range(n_aug)
        if i not in definite and x_aug[i] > tol
    ]

    # Variáveis em zero podem compor bases degeneradas.
    zero_vars = [
        i for i in range(n_aug)
        if abs(x_aug[i]) < tol
    ]

    # Pool ordenado por prioridade para reduzir tentativas ruins de base.
    pool = definite + positive_int + zero_vars

    # Limite de busca para evitar explosão combinatória.
    max_pool = min(len(pool), m + 6)
    pool = pool[:max_pool]

    def _try_basis(indices):
        trial = sorted(indices)
        B = A_full[:, trial]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            # Matriz singular; não pode ser base.
            return None, None, None
        if np.linalg.cond(B) > 1e10:
            # Base numericamente instável; descartamos para evitar cortes ruins.
            return None, None, None
        x_B = B_inv @ b_bar
        # Checagem de factibilidade: x_B >= 0 e coerência com a solução LP atual.
        if (np.all(x_B >= -1e-4) and
                np.allclose(B @ x_B, b_bar, atol=1e-4) and
                np.allclose(x_B, x_aug[trial], atol=1e-4)):
            return trial, B_inv, x_B
        return None, None, None

    for combo in itertools.combinations(pool, m):
        idx, B_inv, x_B = _try_basis(combo)
        if idx is not None:
            return idx, B_inv, x_B

    return None, None, None


# ---------------------------------------------------------------------------
# Cortes fracionários de Gomory (implementação correta via reconstrução de base)
# ---------------------------------------------------------------------------

def gomory_cuts(
    x: np.ndarray,
    model: "MIPModel",
    A_ub_eff: np.ndarray | None,
    b_ub_eff: np.ndarray | None,
    slack: np.ndarray | None = None,
    node_bounds: list | None = None,
    tol: float = 1e-6,
    max_cuts: int = 5,
    cut_details: list | None = None,
) -> list[tuple[np.ndarray, float]]:
    """
        Gera cortes fracionários de Gomory a partir do tableau simplex ótimo do LP.

        Para cada variável inteira básica fracionária x_k:
      tableau row k: a_bar_kj (for all variables in augmented system)
      Gomory cut: - sum_{j} frac(a_bar_kj) * x_orig_j <= - frac(x_B_k)

        Esse corte é globalmente válido (não remove nenhum ponto inteiro factível)
        e é violado pela solução fracionária atual do LP.
    """
    n = len(x)

    # Sem restrições <= efetivas, não há tableau útil para separar Gomory.
    if A_ub_eff is None or b_ub_eff is None or len(A_ub_eff) == 0:
        return []

    m = len(b_ub_eff)

    # Valores de slack da solução atual; reaproveita se vierem do solver.
    if slack is not None and len(slack) == m:
        s = np.asarray(slack, dtype=float)
    else:
        s = b_ub_eff - A_ub_eff @ x

    # Bounds efetivos do nó (globais ou apertados por branching).
    eff_bounds = node_bounds if node_bounds is not None else model.bounds
    lb_arr = np.array([b[0] if b[0] is not None else 0.0 for b in eff_bounds])
    ub_arr = np.array([b[1] if b[1] is not None else 1e30 for b in eff_bounds])

    # Augmented system: [A_ub | I_m] @ [x; s] = b_ub
    # BUT for bounded non-basics at UB, the RHS needs adjustment:
    # b_bar = b - sum_{j non-basic at UB} A[:,j] * ub_j
    x_aug = np.concatenate([x, s])
    A_full = np.hstack([A_ub_eff, np.eye(m)])

    # Identifica variáveis não-básicas no limite superior.
    nonbasic_UB = [j for j in range(n) if abs(x[j] - ub_arr[j]) < tol and ub_arr[j] < 1e29]

    # Ajuste do RHS para variáveis fixadas em UB/LB na forma padrão do tableau.
    b_bar = b_ub_eff.copy()
    for j in nonbasic_UB:
        b_bar = b_bar - A_ub_eff[:, j] * ub_arr[j]
    # Também remove contribuição de não-básicas no LB (quando LB != 0).
    nonbasic_LB = [j for j in range(n) if j not in nonbasic_UB and abs(x[j] - lb_arr[j]) < tol]
    for j in nonbasic_LB:
        if abs(lb_arr[j]) > tol:  # non-zero LB
            b_bar = b_bar - A_ub_eff[:, j] * lb_arr[j]


    # Reconstrói uma base compatível com a solução ótima atual.
    # os objetos que definem o tableau ótimo do simplex:
    #   - basic_idx: quais colunas estão na base,
    #   - B_inv: inversa da base,
    #   - x_B: valores das variáveis básicas.
    basic_idx, B_inv, x_B = _find_basis(A_full, b_bar, x_aug, m, tol)

    if basic_idx is None:
        # Sem base estável/compatível, é mais seguro não gerar cortes.
        return []

    if cut_details is not None:
        full_tableau = B_inv @ A_full   # (m, n+m); tableau completo na base ótima
        frac_rows = [
            (k, bi)
            for k, bi in enumerate(basic_idx)
            if bi < n and model.integrality[bi] > 0
            and tol <= _frac(x_B[k]) <= 1 - tol
        ]
        cut_details.append({
            "type": "gomory_tableau",
            "full_tableau": full_tableau,
            "basic_idx": list(basic_idx),
            "x_B": x_B.copy(),
            "n_orig": n,
            "m_slack": m,
            "frac_rows": frac_rows,
            "var_names": list(model.var_names),
        })

    # Quando uma variável não-básica está no limite superior (UB), faço a troca:
    #   x_j' = ub_j - x_j
    # Assim, no tableau, ela volta a se comportar como uma não-básica em zero.
    # assim consigo aplicar a mesma regra de parte fracionária
    # sem quebrar a validade global do corte.
    #
    # mantenho o corte final em função de x (variável original),
    # mas guardo quem está em UB (nonbasic_UB) para ajustar sinais/coeficientes
    # corretamente na montagem de cut_lhs e cut_rhs.

    cuts = []
    # Cortes de Gomory fazem sentido apenas em linhas básicas ligadas a variáveis inteiras.
    int_basic_rows = [
        (k, bi) for k, bi in enumerate(basic_idx)
        if bi < n and model.integrality[bi] > 0
    ]

    for k, bi in int_basic_rows:
        xB_k = x_B[k]
        f_k = _frac(xB_k)
        if f_k < tol or f_k > 1 - tol:
            # Linha efetivamente inteira; não gera corte útil.
            continue

        # construo a linha k do tableau de Gomory.
        # Faço isso como B^{-1}[k,:] @ A_full, ou seja: pego a k-ésima linha de B^{-1}
        # e projeto sobre todas as colunas do sistema aumentado (originais + slacks).
        # Esse vetor (tableau_row_k) é a matéria-prima dos coeficientes fracionários do corte.
        # Linha k do tableau completo (sobre todas as variáveis aumentadas n+m)
        tableau_row_k = B_inv[k, :] @ A_full   # shape (n+m,)

        #  converto para o corte só nas variáveis originais.
        # Corte de Gomory sobre variáveis ORIGINAIS (colunas 0..n-1)
        # Para não-básicas em LB: coeficiente é -frac(a_kj)
        # Para não-básicas em UB: coeficiente é +frac(-a_kj) = +frac(1-a_kj) se a_kj inteiro, senão -(frac(a_kj))
        # Simplificado para não-básicas com LB=0:
        # Corte de Gomory (na forma <=):
        #   Para não-básica em LB: cut_lhs[j] = -frac(a_kj)
        #   Para não-básica em UB: cut_lhs[j] = +frac(-a_kj)   ← sinal é POSITIVO
        #   RHS = -frac(b_k) + sum_{j em UB} frac(-a_kj)*ub_j
        # Variáveis básicas contribuem 0, pois frac(entrada e_k)=frac(1)=0.
        cut_lhs = np.zeros(n)
        for j in range(n):
            a_kj = tableau_row_k[j]
            if j in nonbasic_UB:
                cut_lhs[j] = _frac(-a_kj)          # contribuição positiva
            else:
                cut_lhs[j] = -_frac(a_kj)          # contribuição não-positiva

        cut_rhs = -f_k
        for j in nonbasic_UB:
            cut_rhs += _frac(-tableau_row_k[j]) * ub_arr[j]

        # Aceita apenas cortes realmente violados pela solução fracionária atual do LP.
        violation = cut_lhs @ x - cut_rhs
        if violation > tol:
            cuts.append((cut_lhs, cut_rhs))
            if cut_details is not None:
                cut_details.append({
                    "type": "gomory",
                    "var_idx": bi,
                    "var_name": model.var_names[bi] if bi < len(model.var_names) else f"x{bi}",
                    "xB_k": float(xB_k),
                    "f_k": float(f_k),
                    "row_k": k,
                    "cut_lhs": cut_lhs.copy(),
                    "cut_rhs": float(cut_rhs),
                    "violation": float(violation),
                })
            if len(cuts) >= max_cuts:
                # Limite de cortes por rodada para controlar custo de separação.
                break

    return cuts


# ---------------------------------------------------------------------------
# Cover cuts (para restrições binárias do tipo knapsack)
# ---------------------------------------------------------------------------

def cover_cuts(
    x: np.ndarray,
    model: "MIPModel",
    tol: float = 1e-6,
) -> list[tuple[np.ndarray, float]]:
    """
    Gera cover cuts para restrições binárias tipo knapsack.

    Para uma restrição knapsack sum_j a_j x_j <= b (a_j, b >= 0),
    uma cover C é um subconjunto com sum_{j in C} a_j > b.
    Cover cut: sum_{j in C} x_j <= |C| - 1.

    Isso é sempre válido para variáveis 0-1.
    """
    cuts = []
    if model.A_ub is None:
        return cuts

    int_idx = set(model.integer_indices)
    n = len(x)

    for row_idx, row in enumerate(model.A_ub):
        b = model.b_ub[row_idx]

        # Mantém apenas restrições tipo knapsack (coeficientes e RHS não-negativos).
        if b < -tol or not all(row >= -tol):
            continue

        # Constrói uma cover gulosa: maiores itens primeiro até exceder a capacidade.
        items = [(j, row[j]) for j in range(n) if row[j] > tol and j in int_idx]
        if not items:
            continue
        items.sort(key=lambda t: t[1], reverse=True)

        cover = []
        total = 0.0
        for j, a in items:
            cover.append(j)
            total += a
            if total > b + tol:
                break

        if total <= b + tol or len(cover) < 2:
            continue

        lhs = np.zeros(n)
        for j in cover:
            lhs[j] = 1.0
        rhs = float(len(cover) - 1)

        # Mantém apenas cortes que realmente eliminam a solução LP atual.
        if lhs @ x > rhs + tol:
            cuts.append((lhs, rhs))

    return cuts
