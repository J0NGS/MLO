"""
Problema 10 - Planejamento de Lotes (Lot Sizing / ULSP) [PLIM]

Uma fabrica planeja producao em T=5 periodos. A cada periodo pode-se
configurar a maquina (custo fixo f_t) e produzir q_t >= 0 unidades.
Estoques s_t >= 0 sao permitidos a custo h_t por unidade por periodo.

Formulacao PLIM (Uncapacitated Lot Sizing - ULS):
    min  sum_t f_t * y_t  +  sum_{t=0}^{T-2} h_t * s_t
    s.t.
    (1)  q_t <= M_t * y_t      para cada t          (producao so com setup)
    (2a) q_0 - s_0 = d_0                             (balanco periodo 0)
    (2b) s_{t-1} + q_t - s_t = d_t   t = 1..T-2    (balanco intermedios)
    (2c) s_{T-2} + q_{T-1} = d_{T-1}               (balanco ultimo)
    (3)  y_t in {0,1},  q_t >= 0,  s_t >= 0        (PLIM)

    M_t = sum_{k=t}^{T-1} d_k   (big-M)

Variaveis (3T-1 no total):
    y_t -> t           (t = 0..T-1, binario)
    q_t -> T + t       (t = 0..T-1, continuo >= 0)
    s_t -> 2T + t      (t = 0..T-2, continuo >= 0; s_{T-1}=0 implicito)

Instancia (T=5 periodos):
    Periodo:     P1    P2    P3    P4    P5
    Setup f:    100   120    90   110   100
    Holding h:    2     3     2     3     2
    Demanda d:   10     8    15     5    12   (total = 50)

Otimo: produzir em P1 e P3, custo = 276.
    P1: q=18 (cobre P1+P2), s1=8
    P3: q=32 (cobre P3+P4+P5), s3=17, s4=12
    Custo = (100+90) + (2*8 + 2*17 + 3*12) = 190 + 86 = 276

Analise de sensibilidade (f3 em [50..170]):
    f3 < 140  -> {P1, P3}, custo = f3 + 186
    f3 = 140  -> empate {P1,P3} = {P1,P5} = 326
    f3 > 140  -> {P1, P5}, custo = 326 (constante)
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult


# ------------------------------------------------------------------
# Default instance
# ------------------------------------------------------------------

T = 5

PERIOD_NAMES = [f"P{t+1}" for t in range(T)]

DEFAULT_SETUP:   list[float] = [100.0, 120.0,  90.0, 110.0, 100.0]
DEFAULT_HOLDING: list[float] = [  2.0,   3.0,   2.0,   3.0,   2.0]
DEFAULT_DEMAND:  list[float] = [ 10.0,   8.0,  15.0,   5.0,  12.0]


# ------------------------------------------------------------------
# Variable index helpers
# ------------------------------------------------------------------

def iy(t: int, n_t: int) -> int:
    return t

def iq(t: int, n_t: int) -> int:
    return n_t + t

def is_(t: int, n_t: int) -> int:
    """Inventory s_t, defined only for t = 0..n_t-2."""
    assert 0 <= t <= n_t - 2, f"s_{t} out of range for T={n_t}"
    return 2 * n_t + t


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    setup:   list[float],
    holding: list[float],
    demand:  list[float],
    period_names: list[str] | None = None,
) -> MIPModel:
    n_t = len(demand)
    if period_names is None:
        period_names = [f"p{t+1}" for t in range(n_t)]

    n_vars = 3 * n_t - 1   # y:n_t  q:n_t  s:n_t-1

    # Objective
    c = [0.0] * n_vars
    for t in range(n_t):
        c[iy(t, n_t)] = float(setup[t])
    for t in range(n_t - 1):
        c[is_(t, n_t)] = float(holding[t])

    # Bounds
    bounds = [(0.0, 1.0)] * n_t          # y_t in [0,1]
    bounds += [(0.0, None)] * n_t        # q_t >= 0
    bounds += [(0.0, None)] * (n_t - 1) # s_t >= 0

    # Integrality: y binary, q and s continuous
    integrality = [2] * n_t + [0] * (2 * n_t - 1)

    # Big-M: q_t - M_t * y_t <= 0
    A_ub, b_ub = [], []
    for t in range(n_t):
        M_t = sum(demand[k] for k in range(t, n_t))
        row = [0.0] * n_vars
        row[iq(t, n_t)] = 1.0
        row[iy(t, n_t)] = -float(M_t)
        A_ub.append(row)
        b_ub.append(0.0)

    # Balance constraints (T equations)
    A_eq, b_eq = [], []

    # t=0: q_0 - s_0 = d_0
    row = [0.0] * n_vars
    row[iq(0, n_t)] =  1.0
    row[is_(0, n_t)] = -1.0
    A_eq.append(row); b_eq.append(float(demand[0]))

    # t=1..T-2: s_{t-1} + q_t - s_t = d_t
    for t in range(1, n_t - 1):
        row = [0.0] * n_vars
        row[is_(t - 1, n_t)] = 1.0
        row[iq(t, n_t)]      = 1.0
        row[is_(t, n_t)]     = -1.0
        A_eq.append(row); b_eq.append(float(demand[t]))

    # t=T-1: s_{T-2} + q_{T-1} = d_{T-1}
    row = [0.0] * n_vars
    row[is_(n_t - 2, n_t)] = 1.0
    row[iq(n_t - 1, n_t)]  = 1.0
    A_eq.append(row); b_eq.append(float(demand[n_t - 1]))

    var_names  = [f"y{period_names[t]}" for t in range(n_t)]
    var_names += [f"q{period_names[t]}" for t in range(n_t)]
    var_names += [f"s{period_names[t]}" for t in range(n_t - 1)]

    return MIPModel(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, integrality=integrality,
        sense="min",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# Warm-start heuristics
# ------------------------------------------------------------------

def single_period_warm_start(
    setup: list[float],
    demand: list[float],
) -> tuple[float, np.ndarray]:
    """Trivial upper bound: y_t=1, q_t=d_t, s_t=0. Cost = sum(f)."""
    n_t = len(demand)
    x = np.zeros(3 * n_t - 1)
    for t in range(n_t):
        x[iy(t, n_t)] = 1.0
        x[iq(t, n_t)] = float(demand[t])
    return float(sum(setup)), x


def silver_meal_warm_start(
    setup: list[float],
    holding: list[float],
    demand: list[float],
) -> tuple[float, np.ndarray]:
    """
    Silver-Meal heuristic: extend each lot while average cost/period decreases.
    """
    n_t = len(demand)
    x = np.zeros(3 * n_t - 1)
    t = 0
    while t < n_t:
        x[iy(t, n_t)] = 1.0
        avg = float(setup[t])
        best_k = 1
        cum_h = 0.0
        for k in range(t + 1, n_t):
            cum_h += demand[k] * sum(holding[j] for j in range(t, k))
            new_avg = (setup[t] + cum_h) / (k - t + 1)
            if new_avg < avg:
                avg = new_avg
                best_k = k - t + 1
            else:
                break
        x[iq(t, n_t)] = float(sum(demand[t:t + best_k]))
        t += best_k

    # recompute inventories
    carry = 0.0
    for t in range(n_t - 1):
        carry = carry + x[iq(t, n_t)] - demand[t]
        x[is_(t, n_t)] = max(0.0, carry)

    cost = (sum(setup[t] * round(x[iy(t, n_t)]) for t in range(n_t))
            + sum(holding[t] * x[is_(t, n_t)] for t in range(n_t - 1)))
    return cost, x


# ------------------------------------------------------------------
# Decoder and printer
# ------------------------------------------------------------------

def decode_solution(
    result: BBResult,
    n_t: int,
) -> tuple[list[int], list[float], list[float]]:
    """Returns (setup_periods, q_list, s_list) where s_list has n_t entries (last=0)."""
    if result.x is None:
        return [], [], []
    setup_periods = [t for t in range(n_t) if result.x[iy(t, n_t)] > 0.5]
    q = [result.x[iq(t, n_t)] for t in range(n_t)]
    s = [result.x[is_(t, n_t)] for t in range(n_t - 1)] + [0.0]
    return setup_periods, q, s


def print_solution(
    result: BBResult,
    setup:   list[float],
    holding: list[float],
    demand:  list[float],
    label: str = "",
    period_names: list[str] | None = None,
):
    n_t = len(demand)
    if period_names is None:
        period_names = [f"P{t+1}" for t in range(n_t)]

    print("\n" + "=" * 68)
    if label:
        print(f"  {label}")
    print(f"Status          : {result.status}")
    print(f"Nos explorados  : {result.nodes_explored}")
    print(f"Tempo (s)       : {result.elapsed:.4f}")

    if result.x is None:
        print("Sem solucao viavel.")
        print("=" * 68)
        return

    sp, q, s = decode_solution(result, n_t)
    total_setup   = sum(setup[t]   for t in sp)
    total_holding = sum(holding[t] * s[t] for t in range(n_t - 1))

    print(f"Custo total     : {result.obj_value:.1f}")
    print(f"  Setup         : {total_setup:.1f}")
    print(f"  Estoque       : {total_holding:.1f}")
    print(f"Periodos setup  : {{{', '.join(period_names[t] for t in sp)}}}")
    print()
    print(f"  {'Per.':<5} {'Setup':>6} {'Dem.':>6} {'Prod.':>8} {'Estq.fin.':>10} {'h*s':>8}")
    print("  " + "-" * 48)
    for t in range(n_t):
        flag = "SIM" if t in sp else "nao"
        hs   = holding[t] * s[t] if t < n_t - 1 else 0.0
        print(f"  {period_names[t]:<5} {flag:>6} {demand[t]:>6.0f} "
              f"{q[t]:>8.1f} {s[t]:>10.1f} {hs:>8.1f}")
    print("=" * 68)


# ------------------------------------------------------------------
# Sensitivity: vary setup cost of one period
# ------------------------------------------------------------------

def sensitivity_analysis(
    f_range: list[float],
    period_idx: int = 2,
    setup_base: list[float] = DEFAULT_SETUP,
    holding: list[float]    = DEFAULT_HOLDING,
    demand:  list[float]    = DEFAULT_DEMAND,
) -> list[tuple[float, float | None, list[int]]]:
    results = []
    for f_val in f_range:
        setup = list(setup_base)
        setup[period_idx] = f_val
        model = build_model(setup, holding, demand)
        g_cost, g_x = silver_meal_warm_start(setup, holding, demand)
        r = BranchAndBound(
            model, strategy="best_first", branching="first_fractional",
            initial_incumbent=g_cost, initial_x=g_x, verbose=False,
        ).solve()
        sp, _, _ = decode_solution(r, len(demand)) if r.x is not None else ([], None, None)
        results.append((f_val, r.obj_value, sp))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    pn = PERIOD_NAMES
    f, h, d = DEFAULT_SETUP, DEFAULT_HOLDING, DEFAULT_DEMAND

    print("=" * 68)
    print("PROBLEMA 10 - Planejamento de Lotes (Lot Sizing ULSP PLIM)")
    print("=" * 68)
    print(f"\n  T={T} periodos | demanda total = {sum(d):.0f}")
    print()
    print(f"  {'Per.':<6} {'f (setup)':>10} {'h (hold)':>10} {'d (dem.)':>10} {'BigM':>8}")
    print("  " + "-" * 50)
    for t in range(T):
        M_t = sum(d[k] for k in range(t, T))
        print(f"  {pn[t]:<6} {f[t]:>10.0f} {h[t]:>10.0f} {d[t]:>10.0f} {M_t:>8.0f}")

    model = build_model(f, h, d, pn)

    g_cost_sp, g_x_sp = single_period_warm_start(f, d)
    g_cost_sm, g_x_sm = silver_meal_warm_start(f, h, d)
    g_cost = min(g_cost_sp, g_cost_sm)
    g_x    = g_x_sm if g_cost_sm <= g_cost_sp else g_x_sp

    print(f"\n  Warm-start periodo-unico : custo = {g_cost_sp:.1f}")
    print(f"  Warm-start Silver-Meal   : custo = {g_cost_sm:.1f}")

    print("\n--- Branch-and-Bound ---")
    bb = BranchAndBound(model, strategy="best_first", branching="first_fractional",
                        initial_incumbent=g_cost, initial_x=g_x)
    r_bb = bb.solve()
    print_solution(r_bb, f, h, d, label="B&B | Instancia original", period_names=pn)

    print("\n--- Branch-and-Cut (cortes de Gomory) ---")
    bc = BranchAndCut(model, strategy="best_first", branching="first_fractional",
                      cut_types=["gomory"],
                      initial_incumbent=g_cost, initial_x=g_x)
    r_bc = bc.solve()
    print_solution(r_bc, f, h, d, label="B&C | Instancia original", period_names=pn)

    print(f"\nB&B nos={r_bb.nodes_explored} | B&C nos={r_bc.nodes_explored}")

    # ---- Sensitivity: vary f3 ----
    print("\n" + "=" * 68)
    print("ANALISE DE SENSIBILIDADE — custo de setup de P3 (f3)")
    print("=" * 68)
    f3_range = [50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0]
    sens = sensitivity_analysis(f3_range, period_idx=2)

    print(f"\n  {'f3':>6}  {'Custo total':>12}  Periodos com setup")
    print("  " + "-" * 52)
    prev = None
    for f3v, total, sp in sens:
        if total is None:
            print(f"  {f3v:>6.0f}  {'inviavel':>12}")
            continue
        pstr  = "{" + ", ".join(pn[t] for t in sp) + "}"
        chg   = " <-- mudanca" if sp != prev and prev is not None else ""
        print(f"  {f3v:>6.0f}  {total:>12.1f}  {pstr}{chg}")
        prev = sp

    print("\nInterpretacao:")
    print("  f3 < 140 -> {P1,P3}: P1 cobre P1+P2 (s1=8), P3 cobre P3+P4+P5")
    print("              custo = f3 + 186  (setup + estoque)")
    print("  f3 = 140 -> empate: {P1,P3} = {P1,P5} = 326")
    print("  f3 > 140 -> {P1,P5}: P1 cobre P1..P4 (maior estoque),")
    print("              P5 produz apenas d5=12. Custo = 326 (fixo)")
