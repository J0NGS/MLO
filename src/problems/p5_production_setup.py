"""
Problema 5 - Planejamento de Producao com Custos Fixos de Setup [PLIM]

Uma fabrica produz 3 chips (A, B, C). Cada chip tem custo fixo de setup
(pago apenas se o chip for produzido) e custo variavel por unidade.

Formulacao PLIM:
    max  sum_k (rev_k - cvar_k)*q_k - sum_k setup_k*y_k
    s.t.
    (1)  q_A + q_B + q_C                <= 60          (producao total)
    (2)  1*q_A + 1.5*q_B + 2*q_C       <= 80          (horas-maquina)
    (3)  q_k <= M_k * y_k               para cada k    (Big-M setup)
    (4)  q_k >= dem_k * y_k             para cada k    (demanda minima)
         q_k >= 0 (continua),  y_k in {0,1}

Variaveis (indices 0..5): [q_A, q_B, q_C, y_A, y_B, y_C]

Big-M escolhido tight por chip:
    M_A = min(prod_cap, hrs_cap / h_A) = min(60, 80) = 60
    M_B = min(60, 80/1.5) = 53.33...
    M_C = min(60, 80/2.0) = 40
Valores menores de M tightnam a relaxacao LP, reduzindo o gap.
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult


# ------------------------------------------------------------------
# Problem data
# ------------------------------------------------------------------

CHIPS = ["A", "B", "C"]

DEFAULT_SETUP   = [200.0, 150.0, 300.0]  # setup cost ($)
DEFAULT_CVAR    = [8.0,   6.0,   12.0]   # variable cost ($/unit)
DEFAULT_REVENUE = [20.0,  18.0,  30.0]   # revenue ($/unit)
DEFAULT_DEMAND  = [10.0,  15.0,  8.0]    # minimum batch if produced
DEFAULT_HOURS   = [1.0,   1.5,   2.0]    # machine hours per unit

PROD_CAP = 60.0   # max total units
HOUR_CAP = 80.0   # max machine hours


# ------------------------------------------------------------------
# Tight Big-M per chip
# ------------------------------------------------------------------

def compute_big_m(hours: list[float], prod_cap: float, hour_cap: float) -> list[float]:
    """Upper bound on q_k from production and hours constraints."""
    return [min(prod_cap, hour_cap / h) for h in hours]


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    setup: list[float] = DEFAULT_SETUP,
    cvar: list[float] = DEFAULT_CVAR,
    revenue: list[float] = DEFAULT_REVENUE,
    demand_min: list[float] = DEFAULT_DEMAND,
    hours: list[float] = DEFAULT_HOURS,
    prod_cap: float = PROD_CAP,
    hour_cap: float = HOUR_CAP,
) -> MIPModel:
    n = len(CHIPS)           # 3 chips
    n_vars = 2 * n           # q_A..q_C, y_A..y_C

    def iq(k): return k          # index of q_k
    def iy(k): return n + k      # index of y_k

    M = compute_big_m(hours, prod_cap, hour_cap)

    # Objective: max sum_k margin_k*q_k - sum_k setup_k*y_k
    margin = [revenue[k] - cvar[k] for k in range(n)]
    c = [0.0] * n_vars
    for k in range(n):
        c[iq(k)] = margin[k]
        c[iy(k)] = -setup[k]

    bounds = [(0.0, None)] * n + [(0.0, 1.0)] * n
    integrality = [0] * n + [2] * n   # q continuous, y binary

    A_ub, b_ub = [], []

    # (1) Production total: sum_k q_k <= prod_cap
    row = [0.0] * n_vars
    for k in range(n):
        row[iq(k)] = 1.0
    A_ub.append(row); b_ub.append(prod_cap)

    # (2) Machine hours: sum_k h_k * q_k <= hour_cap
    row = [0.0] * n_vars
    for k in range(n):
        row[iq(k)] = hours[k]
    A_ub.append(row); b_ub.append(hour_cap)

    # (3) Big-M: q_k - M_k * y_k <= 0
    for k in range(n):
        row = [0.0] * n_vars
        row[iq(k)] = 1.0
        row[iy(k)] = -M[k]
        A_ub.append(row); b_ub.append(0.0)

    # (4) Min demand: -q_k + dem_k * y_k <= 0  (i.e. q_k >= dem_k * y_k)
    for k in range(n):
        row = [0.0] * n_vars
        row[iq(k)] = -1.0
        row[iy(k)] = demand_min[k]
        A_ub.append(row); b_ub.append(0.0)

    var_names = [f"q{c}" for c in CHIPS] + [f"y{c}" for c in CHIPS]

    return MIPModel(
        c=c, A_ub=A_ub, b_ub=b_ub,
        bounds=bounds, integrality=integrality,
        sense="max",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# Printer
# ------------------------------------------------------------------

def print_solution(
    result: BBResult,
    setup: list[float] = DEFAULT_SETUP,
    cvar: list[float] = DEFAULT_CVAR,
    revenue: list[float] = DEFAULT_REVENUE,
    label: str = "",
):
    n = len(CHIPS)
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

    print(f"Lucro otimo    : ${result.obj_value:.2f}")
    print()
    print(f"  {'Chip':<6} {'Prod?':<7} {'Qtd':>8} {'Receita':>10} {'C.var':>10} {'Setup':>8} {'Lucro':>10}")
    print("  " + "-" * 59)

    total_revenue = 0.0
    total_cvar = 0.0
    total_setup = 0.0
    for k in range(n):
        q = result.x[k]
        y = round(result.x[n + k])
        rev_k = revenue[k] * q
        cv_k  = cvar[k] * q
        s_k   = setup[k] * y
        lucro_k = rev_k - cv_k - s_k
        total_revenue += rev_k
        total_cvar += cv_k
        total_setup += s_k
        flag = "SIM" if y == 1 else "nao"
        print(f"  {CHIPS[k]:<6} {flag:<7} {q:>8.2f} {rev_k:>10.2f} {cv_k:>10.2f} {s_k:>8.2f} {lucro_k:>10.2f}")

    print("  " + "-" * 59)
    total_profit = total_revenue - total_cvar - total_setup
    print(f"  {'TOTAL':<13} {'':>8} {total_revenue:>10.2f} {total_cvar:>10.2f} {total_setup:>8.2f} {total_profit:>10.2f}")
    print("=" * 65)


# ------------------------------------------------------------------
# Sensitivity analysis on setup cost of C
# ------------------------------------------------------------------

def sensitivity_analysis(
    setup_C_range: list[float],
    **model_kwargs,
) -> list[tuple[float, float | None, list[float] | None]]:
    """
    Run B&B for each setup_C value.
    Returns list of (setup_C, optimal_profit, x_vector).
    """
    results = []
    for sc in setup_C_range:
        setup = list(DEFAULT_SETUP)
        setup[2] = sc
        model = build_model(setup=setup, **model_kwargs)
        solver = BranchAndBound(model, strategy="best_first",
                                branching="most_infeasible", verbose=False)
        r = solver.solve()
        results.append((sc, r.obj_value, r.x))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    M = compute_big_m(DEFAULT_HOURS, PROD_CAP, HOUR_CAP)

    print("=" * 65)
    print("PROBLEMA 5 - Planejamento de Producao com Custos de Setup")
    print("=" * 65)
    print(f"Capacidade: {PROD_CAP:.0f} unidades, {HOUR_CAP:.0f} horas-maquina")
    print(f"Big-M (tight por chip): A={M[0]:.2f}, B={M[1]:.2f}, C={M[2]:.2f}")
    print()
    print(f"  {'Chip':<5} {'Margem':>8} {'Setup':>8} {'Dem.min':>8} {'Horas':>7}")
    margin = [DEFAULT_REVENUE[k] - DEFAULT_CVAR[k] for k in range(3)]
    for k in range(3):
        print(f"  {CHIPS[k]:<5} {margin[k]:>8.2f} {DEFAULT_SETUP[k]:>8.2f} {DEFAULT_DEMAND[k]:>8.2f} {DEFAULT_HOURS[k]:>7.2f}")

    model = build_model()

    print("\n--- Branch-and-Bound (ramifica apenas em y_k) ---")
    bb = BranchAndBound(model, strategy="best_first", branching="most_infeasible")
    r_bb = bb.solve()
    print_solution(r_bb, label="B&B | Instancia original")

    print("\n--- Branch-and-Cut (cortes de Gomory em y_k fracionarios) ---")
    bc = BranchAndCut(model, strategy="best_first", branching="most_infeasible",
                      cut_types=["gomory"])
    r_bc = bc.solve()
    print_solution(r_bc, label="B&C | Instancia original")

    print(f"\nB&B nos={r_bb.nodes_explored} | B&C nos={r_bc.nodes_explored}")

    # ---- Sensitivity analysis: setup_C from 100 to 400 in steps of 50 ----
    print("\n" + "=" * 65)
    print("ANALISE DE SENSIBILIDADE — custo de setup do Chip C")
    print("=" * 65)
    setup_C_values = [float(v) for v in range(100, 401, 50)]
    sens = sensitivity_analysis(setup_C_values)

    print(f"\n  {'Setup C':>9} {'Lucro otimo':>13} {'Chip C prod?':>13} {'qtd C':>8}")
    print("  " + "-" * 47)
    prev_decision = None
    for sc, profit, x in sens:
        if profit is None:
            print(f"  {sc:>9.0f} {'infeasible':>13}")
            continue
        y_C = round(x[5]) if x is not None else 0
        q_C = x[2] if x is not None else 0.0
        decision = "SIM" if y_C else "nao"
        change = " <-- mudanca" if decision != prev_decision and prev_decision is not None else ""
        print(f"  {sc:>9.0f} {profit:>13.2f} {decision:>13} {q_C:>8.2f}{change}")
        prev_decision = decision

    print("\nInterpretacao:")
    print("  Setup C < 200 -> produzir apenas C (maior margem por hora)")
    print("  Setup C >= 200 -> produzir apenas A (menor setup, producao max)")
    print("  Ponto de indiferenca: setup C = 200 (lucro = 520 em ambos os casos)")
