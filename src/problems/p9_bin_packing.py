"""
Problema 9 - Empacotamento de Caixas (Bin Packing) via Geracao de Colunas e Branch-and-Price

Formulacao por Geracao de Colunas (modelo de padroes):

  Padroes: subconjuntos P de itens com sum_{i in P} w_i <= B
  Variavel z_p = 1 se o padrao p e usado, 0 c.c.

  PMR (Problema Mestre Restrito — relaxacao LP):
    min  sum_p z_p
    s.t. sum_p a_{ip} * z_p = 1    para cada item i   (set partition)
         z_p >= 0

  Subproblema de precificacao (mochila 0-1):
    max  sum_i pi_i * a_i
    s.t. sum_i w_i * a_i <= B,  a_i in {0,1}
  Custo reduzido do candidato: rc = 1 - max_knap.  Adiciona se rc < 0.

  Branch-and-Price (B&P):
    B&B sobre PMR inteiro; em cada no re-resolve por CG.
    Ramifica em z_p mais fracionario (z_p <= floor | z_p >= ceil).

Instancia (6 itens, B=10):
  Pesos: [6, 4, 4, 3, 3, 2]   soma = 22
  LB trivial: ceil(22/10) = 3 caixas
  Otimo: 3 caixas  {I1,I2} {I3,I4,I5} {I6}
"""
from __future__ import annotations
import math
import sys, os
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ------------------------------------------------------------------
# Instancia padrao
# ------------------------------------------------------------------

N_ITEMS = 6
ITEM_NAMES = [f"I{i+1}" for i in range(N_ITEMS)]
DEFAULT_WEIGHTS: list[float] = [6.0, 4.0, 4.0, 3.0, 3.0, 2.0]
DEFAULT_CAPACITY: float = 10.0


# ------------------------------------------------------------------
# Auxiliares de padroes
# ------------------------------------------------------------------

def pattern_label(pat: tuple, item_names: list[str]) -> str:
    return "+".join(item_names[i] for i in sorted(pat))


def generate_all_feasible_patterns(weights: list[float], capacity: float) -> list[tuple]:
    """Enumera todos os padroes viaveis por forca bruta (2^n subconjuntos)."""
    n = len(weights)
    patterns = []
    for mask in range(1, 1 << n):
        items = tuple(i for i in range(n) if mask & (1 << i))
        if sum(weights[i] for i in items) <= capacity + 1e-9:
            patterns.append(items)
    return patterns


# ------------------------------------------------------------------
# Subproblema de precificacao: mochila 0-1 (DP exato)
# ------------------------------------------------------------------

def knapsack_dp(
    profits: list[float],
    weights: list[float],
    capacity: float,
) -> tuple[float, list[int]]:
    """
    Mochila 0-1 por programacao dinamica.
    Assume capacidade e pesos inteiros (arredondados internamente).
    Retorna (valor_maximo, lista_de_itens_selecionados).
    """
    n = len(profits)
    cap = int(round(capacity))
    int_w = [int(round(w)) for w in weights]

    # dp[i][c] = max lucro usando itens 0..i-1 com capacidade c
    dp = np.zeros((n + 1, cap + 1))
    for i in range(1, n + 1):
        w = int_w[i - 1]
        p = max(profits[i - 1], 0.0)   # lucro negativo: nao ajuda
        dp[i] = dp[i - 1].copy()
        for c in range(w, cap + 1):
            val = dp[i - 1][c - w] + p
            if val > dp[i][c] + 1e-10:
                dp[i][c] = val

    # Backtrack
    items: list[int] = []
    c = cap
    for i in range(n, 0, -1):
        w = int_w[i - 1]
        if w <= c and dp[i][c] > dp[i - 1][c] + 1e-10:
            items.append(i - 1)
            c -= w

    return float(dp[n][cap]), items


# ------------------------------------------------------------------
# LP do PMR
# ------------------------------------------------------------------

def solve_pmr_lp(
    patterns: list[tuple],
    n_items: int,
    lb_z: list[float] | None = None,
    ub_z: list[float] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """
    Resolve a relaxacao LP do PMR (formulacao set partition):

      min 1^T z  s.t.  A z = 1,  z >= 0

    Cada item deve aparecer em EXATAMENTE um padrao ativo (=, nao >=).
    Garante que a solucao inteira do B&P seja um empacotamento valido.

    Duals: pi = result.eqlin.marginals  (sem restricao de sinal)
      - pi_i e o preco-sombra de "cobrir o item i exatamente uma vez"
      - Custo reduzido do padrao p: rc_p = 1 - sum_i a_{ip} pi_i
      - Adiciona p se rc_p < 0  (equivalente: max_knap > 1)

    Retorna (z_vals, pi, obj) ou (None, None, None) se infeasivel.
    """
    n_p = len(patterns)

    # A[i, p] = 1 se item i esta no padrao p
    A = np.zeros((n_items, n_p))
    for p_idx, pat in enumerate(patterns):
        for i in pat:
            A[i, p_idx] = 1.0

    bounds = []
    for p in range(n_p):
        lb = lb_z[p] if lb_z is not None and p < len(lb_z) else 0.0
        ub = ub_z[p] if ub_z is not None and p < len(ub_z) else None
        bounds.append((lb, ub))

    # set partition: A z = 1  (igualdade, nao desigualdade)
    result = linprog(np.ones(n_p), A_eq=A, b_eq=np.ones(n_items),
                     bounds=bounds, method="highs")

    if result.status != 0:
        return None, None, None

    duals = np.asarray(result.eqlin.marginals)  # sem restricao de sinal
    return result.x, duals, float(result.fun)


# ------------------------------------------------------------------
# Loop de geracao de colunas
# ------------------------------------------------------------------

def column_generation(
    weights: list[float],
    capacity: float,
    patterns: list[tuple] | None = None,
    lb_z: list[float] | None = None,
    ub_z: list[float] | None = None,
    verbose: bool = True,
) -> tuple[list[tuple], np.ndarray | None, np.ndarray | None, float | None]:
    """
    Geracao de colunas para o PMR de bin packing.

    Cada iteracao:
      1. Resolve LP do PMR com os padroes atuais
      2. Extrai duais pi_i
      3. Resolve subproblema de mochila: max sum_i pi_i a_i s.t. sum_i w_i a_i <= B
      4. Se max > 1 (rc < 0): adiciona novo padrao e repete
         Senao: LP e otimo (sem coluna de custo reduzido negativo)

    Inicializa com padroes singleton se nenhum for fornecido.
    Retorna (patterns, z_vals, duals, obj) ou (None, None, None, None) se infeasivel.
    """
    n = len(weights)
    inames = [f"I{i+1}" for i in range(n)]

    if patterns is None:
        patterns = [(i,) for i in range(n)]   # singletons: 1 item por caixa
    else:
        patterns = list(patterns)

    # Garante lb_z/ub_z com comprimento suficiente
    if lb_z is not None:
        lb_z = list(lb_z) + [0.0] * max(0, len(patterns) - len(lb_z))
    if ub_z is not None:
        ub_z = list(ub_z) + [None] * max(0, len(patterns) - len(ub_z))

    for iteration in range(1, 300):
        z, duals, obj = solve_pmr_lp(patterns, n, lb_z, ub_z)

        if z is None:
            if verbose:
                print(f"    Iter {iteration:2d}: PMR infeasivel")
            return None, None, None, None

        # Subproblema de precificacao (mochila 0-1)
        max_val, new_items = knapsack_dp(list(duals), weights, capacity)
        rc = 1.0 - max_val

        if verbose:
            pi_str = " ".join(f"{d:.4f}" for d in duals)
            cand_str = ("+".join(inames[i] for i in sorted(new_items))
                        if new_items else "vazio")
            print(f"    Iter {iteration:2d}:  obj={obj:.4f} | "
                  f"pi=[{pi_str}] | "
                  f"mochila={max_val:.4f} rc={rc:+.4f} | "
                  f"candidato={{{cand_str}}} | "
                  f"padroes={len(patterns)}")

        if rc >= -1e-6:
            if verbose:
                print(f"    => LP otimo: {obj:.6f}  "
                      f"({len(patterns)} padroes, {iteration} iteracoes)")
            break

        new_pat = tuple(sorted(new_items))
        if new_pat in patterns:
            if verbose:
                print(f"    => Padrao candidato ja existe; parando")
            break

        patterns.append(new_pat)
        if lb_z is not None:
            lb_z = lb_z + [0.0]
        if ub_z is not None:
            ub_z = ub_z + [None]

    return patterns, z, duals, obj


# ------------------------------------------------------------------
# Branch-and-Price
# ------------------------------------------------------------------

class BranchAndPrice:
    """
    Branch-and-Price para bin packing.

    Em cada no da arvore:
      - Resolve o PMR por geracao de colunas (LP)
      - Se fracionario: ramifica em z_p mais fracionario
        * Ramo down: z_p <= floor(val)   (exclui ou limita o padrao)
        * Ramo up:   z_p >= ceil(val)    (forca o uso do padrao)
      - Se inteiro: atualiza incumbente

    Para z_p in (0,1): branching equivale a z_p=0 (excluir) ou z_p=1 (incluir).
    """

    def __init__(self, weights: list[float], capacity: float, verbose: bool = True):
        self.weights = weights
        self.capacity = capacity
        self.n = len(weights)
        self.verbose = verbose
        self.n_nodes = 0
        self.best_obj = math.inf
        self.best_patterns: list[tuple] | None = None
        self.best_z: np.ndarray | None = None

    def solve(self) -> tuple[float, list[tuple] | None, np.ndarray | None]:
        """Executa B&P. Retorna (obj_inteiro, patterns, z)."""
        init_patterns = [(i,) for i in range(self.n)]
        self._branch(init_patterns, lb_z=None, ub_z=None, depth=0)
        return self.best_obj, self.best_patterns, self.best_z

    def _branch(
        self,
        patterns: list[tuple],
        lb_z: list[float] | None,
        ub_z: list[float] | None,
        depth: int,
    ) -> None:
        self.n_nodes += 1
        node_id = self.n_nodes
        indent = "  " * depth
        inames = [f"I{i+1}" for i in range(self.n)]

        # Geracao de colunas neste no
        patterns, z, _, obj = column_generation(
            self.weights, self.capacity,
            list(patterns), lb_z, ub_z,
            verbose=False,
        )

        if patterns is None:
            if self.verbose:
                print(f"{indent}[No {node_id}] Infeasivel")
            return

        if self.verbose:
            print(f"{indent}[No {node_id}] LP={obj:.4f} | {len(patterns)} padroes")

        # Poda por bound
        if obj >= self.best_obj - 1e-6:
            if self.verbose:
                print(f"{indent}  Podado ({obj:.4f} >= incumbente {self.best_obj:.0f})")
            return

        # Verifica integralidade
        frac = [
            (p, z[p])
            for p in range(len(z))
            if 1e-5 < (z[p] - math.floor(z[p])) < 1 - 1e-5
        ]

        if not frac:
            int_obj = float(sum(round(z[p]) for p in range(len(z))))
            if self.verbose:
                print(f"{indent}  Solucao inteira: {int_obj:.0f} caixas  [incumbente atualizado]")
            if int_obj < self.best_obj - 1e-6:
                self.best_obj = int_obj
                self.best_patterns = list(patterns)
                self.best_z = z.copy()
            return

        # Ramifica na variavel mais fracionaria
        def frac_part(v: float) -> float:
            return v - math.floor(v)

        branch_p, val = max(frac, key=lambda t: min(frac_part(t[1]), 1.0 - frac_part(t[1])))
        floor_val = math.floor(val)
        ceil_val = math.ceil(val)

        if self.verbose:
            pat_str = pattern_label(patterns[branch_p], inames)
            print(f"{indent}  Ramifica z[{{{pat_str}}}]={val:.4f}: "
                  f"<= {floor_val} (excluir) | >= {ceil_val} (incluir)")

        n_p = len(patterns)

        def extend(base: list | None, length: int, default) -> list:
            b = list(base) if base else [default] * length
            while len(b) < length:
                b.append(default)
            return b

        # Ramo down: z_p <= floor_val
        ub_d = extend(ub_z, n_p, None)
        ub_d[branch_p] = float(floor_val)
        self._branch(list(patterns), extend(lb_z, n_p, 0.0), ub_d, depth + 1)

        # Ramo up: z_p >= ceil_val
        lb_u = extend(lb_z, n_p, 0.0)
        lb_u[branch_p] = float(ceil_val)
        self._branch(list(patterns), lb_u, extend(ub_z, n_p, None), depth + 1)


# ------------------------------------------------------------------
# FFD heuristica
# ------------------------------------------------------------------

def ffd_assignment(weights: list[float], capacity: float) -> tuple[dict[int, int], int]:
    """First-Fit Decreasing. Retorna (item->bin_map, n_bins)."""
    order = sorted(range(len(weights)), key=lambda i: -weights[i])
    loads: list[float] = []
    item_bin: dict[int, int] = {}
    for i in order:
        for j, load in enumerate(loads):
            if load + weights[i] <= capacity + 1e-9:
                loads[j] += weights[i]
                item_bin[i] = j
                break
        else:
            item_bin[i] = len(loads)
            loads.append(float(weights[i]))
    return item_bin, len(loads)


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

def print_lp_solution(
    patterns: list[tuple],
    z: np.ndarray,
    obj: float,
    weights: list[float],
    capacity: float,
    item_names: list[str],
) -> None:
    print(f"\n  Relaxacao LP do PMR  (obj = {obj:.6f})")
    print(f"  {len(patterns)} padroes gerados pela CG:")
    print(f"\n  {'#':<4} {'Padrao':<22} {'Peso':>5}  {'z_p':>8}  Status")
    print("  " + "-" * 50)
    for p, (pat, zval) in enumerate(zip(patterns, z)):
        label = "{" + "+".join(item_names[i] for i in sorted(pat)) + "}"
        w = sum(weights[i] for i in pat)
        fp = zval - math.floor(zval)
        if zval < 1e-6:
            status = "zero"
        elif 1e-5 < fp < 1 - 1e-5:
            status = f"FRAC ({fp:.4f})"
        else:
            status = "INTEIRO"
        print(f"  p{p+1:<3} {label:<22} {w:>5.0f}  {zval:>8.4f}  {status}")
    lb = math.ceil(obj - 1e-9)
    print(f"\n  LB da relaxacao LP: {obj:.4f}  =>  ceil = {lb} caixas")
    print(f"  Arredondamento ceil(z_p): {int(sum(math.ceil(v - 1e-9) for v in z))} caixas")


def print_integer_solution(
    patterns: list[tuple],
    z: np.ndarray,
    obj: float,
    weights: list[float],
    capacity: float,
    item_names: list[str],
    label: str = "",
) -> None:
    print(f"\n{'=' * 62}")
    if label:
        print(f"  {label}")
    print(f"  Caixas usadas : {obj:.0f}")
    print()
    bin_num = 1
    for p in range(len(z)):
        k = int(round(z[p]))
        if k < 1:
            continue
        pat = patterns[p]
        w = sum(weights[i] for i in pat)
        items_str = ", ".join(
            f"{item_names[i]}({weights[i]:.0f})" for i in sorted(pat)
        )
        for _ in range(k):
            print(f"    Caixa {bin_num}: [{items_str}]  "
                  f"carga={w:.0f}/{capacity:.0f}")
            bin_num += 1
    print("=" * 62)


# ------------------------------------------------------------------
# Analise de sensibilidade (CG-only para velocidade)
# ------------------------------------------------------------------

def sensitivity_analysis(
    cap_range: list[float],
    weights: list[float] = DEFAULT_WEIGHTS,
) -> list[tuple[float, int | None, int]]:
    """Varia B: roda CG (LB = ceil do LP) e FFD (UB heuristico)."""
    results = []
    for B in cap_range:
        _, z, _, obj = column_generation(weights, B, verbose=False)
        lb_cg = math.ceil(obj - 1e-9) if obj is not None else None
        _, ffd_n = ffd_assignment(weights, B)
        results.append((B, lb_cg, ffd_n))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    W  = DEFAULT_WEIGHTS
    B  = DEFAULT_CAPACITY
    fn = ITEM_NAMES
    n  = N_ITEMS
    total_w = sum(W)

    print("=" * 62)
    print("PROBLEMA 9 - Bin Packing via Geracao de Colunas + B&P")
    print("=" * 62)
    print(f"\n  {n} itens | capacidade B = {B:.0f}")
    print(f"  Pesos : {W}")
    print(f"  Soma  : {total_w:.0f}")
    print(f"  LB trivial: ceil({total_w:.0f}/{B:.0f}) = {math.ceil(total_w / B)}")

    all_pats = generate_all_feasible_patterns(W, B)
    print(f"  Padroes viaveis (enumeracao): {len(all_pats)}")

    # ---- FFD ----
    item_bin, ffd_n = ffd_assignment(W, B)
    print(f"\n  Heuristica FFD: {ffd_n} caixas")
    for j in sorted(set(item_bin.values())):
        items_in = sorted(i for i, b in item_bin.items() if b == j)
        load = sum(W[i] for i in items_in)
        s = ", ".join(f"{fn[i]}({W[i]:.0f})" for i in items_in)
        print(f"    Caixa {j+1}: [{s}] = {load:.0f}")

    # ================================================================
    # Geracao de Colunas - relaxacao LP
    # ================================================================
    print("\n" + "=" * 62)
    print("GERACAO DE COLUNAS - relaxacao LP do PMR")
    print("=" * 62)
    print("\n  Padroes iniciais: singletons (1 item por caixa)")
    for i in range(n):
        print(f"    p{i+1}: {{{fn[i]}}}  peso={W[i]:.0f}")
    print()

    lp_patterns, z_lp, duals_lp, obj_lp = column_generation(W, B, verbose=True)

    print_lp_solution(lp_patterns, z_lp, obj_lp, W, B, fn)

    # ================================================================
    # Branch-and-Price
    # ================================================================
    print("\n" + "=" * 62)
    print("BRANCH-AND-PRICE")
    print("=" * 62)
    print()

    bp = BranchAndPrice(W, B, verbose=True)
    bp_obj, bp_pats, bp_z = bp.solve()

    if bp_pats is not None:
        print_integer_solution(
            bp_pats, bp_z, bp_obj, W, B, fn,
            label="B&P | Solucao otima inteira",
        )

    # ================================================================
    # Comparacao
    # ================================================================
    print("\n" + "=" * 62)
    print("COMPARACAO")
    print("=" * 62)
    lb_lp    = math.ceil(obj_lp - 1e-9)
    rounded  = int(sum(math.ceil(v - 1e-9) for v in z_lp))
    print(f"  Relaxacao LP (CG)       : {obj_lp:.4f}")
    print(f"  LB = ceil(LP)           : {lb_lp} caixas")
    print(f"  Arredondamento ceil(z_p): {rounded} caixas  "
          f"({'otimo' if rounded == bp_obj else 'subotimo'})")
    print(f"  FFD heuristica          : {ffd_n} caixas  "
          f"({'otimo' if ffd_n == bp_obj else 'subotimo'})")
    print(f"  B&P (exato)             : {bp_obj:.0f} caixas")
    print(f"  Nos B&P                 : {bp.n_nodes}")

    # ================================================================
    # Analise de sensibilidade
    # ================================================================
    print("\n" + "=" * 62)
    print("ANALISE DE SENSIBILIDADE - capacidade B")
    print("=" * 62)

    cap_range = [float(v) for v in range(6, 14)]
    sens = sensitivity_analysis(cap_range, W)

    print(f"\n  {'B':>4}  {'LB=ceil(22/B)':>14}  {'CG(ceil LB)':>12}  {'FFD':>5}  {'FFD=opt?':>9}")
    print("  " + "-" * 55)
    prev = None
    for B_val, cg_val, ffd_val in sens:
        lb_triv = math.ceil(total_w / B_val)
        marker = " <-- muda" if cg_val != prev and prev is not None else ""
        ffd_ok = "SIM" if ffd_val == cg_val else "NAO *"
        cg_str = str(cg_val) if cg_val is not None else "?"
        print(f"  {B_val:>4.0f}  {lb_triv:>14}  {cg_str:>12}  {ffd_val:>5}  {ffd_ok:>9}{marker}")
        prev = cg_val

    print("\n  * CG(ceil LB) e o teto da relaxacao LP - upper bound valido.")
    print("  * FFD subotimo em B=11: da 3 caixas, otimo e 2.")
    print("\nInterpretacao:")
    print(f"  B =  6, 7  -> 4 caixas  (3*7=21 < 22, impossivel em 3)")
    print(f"  B =  8-10  -> 3 caixas  ex: {{I1,I2}}, {{I3,I4,I5}}, {{I6}}")
    print(f"  B = 11-13  -> 2 caixas  ex: {{I1,I4,I6}}, {{I2,I3,I5}}")
