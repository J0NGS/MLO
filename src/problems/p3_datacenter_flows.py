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

Estratégia de B&B:
    - Bound de cada nó: mochila fracionária greedy (razão prioridade/banda),
      usando a capacidade de banda REMANESCENTE (variáveis já fixadas
      em 1 consomem capacidade; variáveis em 0 são excluídas).
    - Buffer: usado apenas para poda de inviabilidade (não entra no bound).
    - Ramificação: segue a ordem greedy (maior razão p/b primeiro).
    - Busca best-first (expande o nó com maior bound).

Gap de otimalidade:
    gap = 100 * (best_bound - incumbente) / |incumbente|
    Ao final, se o B&B prova otimalidade, gap = 0%.
"""
from __future__ import annotations
import heapq
import random
import time
from dataclasses import dataclass
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndCut, BBResult
from core.relaxation import solve_relaxation


# ------------------------------------------------------------------
# Model builder (used by B&C)
# ------------------------------------------------------------------

def build_model(
    banda: list[float],
    buffer: list[float],
    priority: list[float],
    banda_cap: float,
    buffer_cap: float,
) -> MIPModel:
    n = len(banda)
    return MIPModel(
        c=list(priority),
        A_ub=[list(banda), list(buffer)],
        b_ub=[float(banda_cap), float(buffer_cap)],
        bounds=[(0.0, 1.0)] * n,
        integrality=[2] * n,
        sense="max",
        var_names=[f"F{j+1}" for j in range(n)],
    )


# ------------------------------------------------------------------
# Fractional knapsack bound (greedy by priority/banda ratio)
# ------------------------------------------------------------------

def fractional_knapsack_bound(
    banda: list[float],
    priority: list[float],
    rem_banda: float,
    free_items: list[int],
) -> float:
    """
    Greedy fractional knapsack upper bound over `free_items`,
    using `rem_banda` as remaining bandwidth capacity.
    Items are assumed already sorted by priority/banda ratio descending.
    Returns the fractional bound contribution from free items.
    """
    cap = rem_banda
    bound = 0.0
    for j in free_items:
        if cap <= 1e-9:
            break
        frac = min(1.0, cap / banda[j])
        bound += frac * priority[j]
        cap -= frac * banda[j]
    return bound


# ------------------------------------------------------------------
# Greedy integer solution (warm-start incumbent)
# ------------------------------------------------------------------

def greedy_integer_solution(
    banda: list[float],
    buffer: list[float],
    priority: list[float],
    banda_cap: float,
    buffer_cap: float,
) -> tuple[float, np.ndarray]:
    """
    Integer greedy solution by priority/banda ratio.
    Respects both constraints (no fractional items).
    Returns (total_priority, x_vector).
    """
    order = sorted(range(len(banda)),
                   key=lambda j: priority[j] / banda[j], reverse=True)
    rem_b = float(banda_cap)
    rem_buf = float(buffer_cap)
    x = np.zeros(len(banda))
    total = 0.0
    for j in order:
        if banda[j] <= rem_b + 1e-9 and buffer[j] <= rem_buf + 1e-9:
            x[j] = 1.0
            rem_b   -= banda[j]
            rem_buf -= buffer[j]
            total   += priority[j]
    return total, x


# ------------------------------------------------------------------
# KnapsackBnB — B&B with FK bounding + most-infeasible branching
# ------------------------------------------------------------------

@dataclass
class KnapsackResult:
    """Result of KnapsackBnB.solve()."""
    status: str          # "optimal" or "timeout"
    obj_value: float     # best integer incumbent found
    x: np.ndarray        # solution vector
    root_fk_bound: float # FK bound at root (initial upper bound)
    best_bound: float    # tightest proved upper bound at termination
    gap_pct: float       # 100*(best_bound - obj)/|obj|; 0.0 if optimal
    nodes_explored: int
    elapsed: float


class KnapsackBnB:
    """
    Best-first Branch-and-Bound for binary 2D knapsack.

    Bounding:   greedy fractional knapsack by priority/banda ratio applied to
                the FREE variables at each node, using remaining bandwidth.
                Buffer constraint is enforced only for feasibility.
    Branching:  most-infeasible — branch on the FK break item (the unique
                fractional variable in the FK relaxation at the current node).
    Warm-start: accepts initial_incumbent and initial_x.
    """

    def __init__(
        self,
        banda: list[float],
        buffer: list[float],
        priority: list[float],
        banda_cap: float,
        buffer_cap: float,
        initial_incumbent: float = 0.0,
        initial_x: np.ndarray | None = None,
        time_limit: float = 300.0,
    ):
        self.banda = list(banda)
        self.buffer = list(buffer)
        self.priority = list(priority)
        self.banda_cap = float(banda_cap)
        self.buffer_cap = float(buffer_cap)
        self.n = len(banda)
        self.time_limit = time_limit

        # Greedy sort order: descending priority/banda ratio
        self.order = sorted(range(self.n),
                            key=lambda j: self.priority[j] / self.banda[j],
                            reverse=True)

        self.incumbent = float(initial_incumbent)
        self.best_x = (np.array(initial_x, dtype=float)
                       if initial_x is not None else np.zeros(self.n))

    def _fk(self, rem_banda: float, cur_prio: float,
            free_bits: int) -> tuple[float, int | None]:
        """
        FK relaxation over free items encoded as bitmask `free_bits`.
        Iterates in greedy ratio order (self.order), skipping fixed items.
        Returns (bound, break_j): total FK upper bound and the index of the
        first fractional item, or None when every free item fits fully.
        """
        cap   = rem_banda
        bound = cur_prio
        for j in self.order:
            if not (free_bits >> j & 1):
                continue          # item already fixed, skip
            if cap <= 1e-9:
                break             # bandwidth gone, remaining items get x=0
            if self.banda[j] <= cap + 1e-9:
                bound += self.priority[j]
                cap   -= self.banda[j]
            else:
                bound += self.priority[j] * cap / self.banda[j]
                return bound, j   # break item (unique fractional var)
        return bound, None        # FK solution is integer

    def _update_incumbent(self, cur_p: float, rem_b: float, rem_buf: float,
                          free_bits: int, sel_bits: int) -> None:
        """
        FK relaxation gave no break item → take every free item that fits
        BOTH constraints (greedy), update incumbent if improved.
        """
        p, rb, rbuf, bits = cur_p, rem_b, rem_buf, sel_bits
        for j in self.order:
            if not (free_bits >> j & 1):
                continue
            if self.banda[j] <= rb + 1e-9 and self.buffer[j] <= rbuf + 1e-9:
                p    += self.priority[j]
                rb   -= self.banda[j]
                rbuf -= self.buffer[j]
                bits |= (1 << j)
        if p > self.incumbent + 1e-8:
            self.incumbent = p
            x_new = np.zeros(self.n)
            for j in range(self.n):
                if bits >> j & 1:
                    x_new[j] = 1.0
            self.best_x = x_new

    def solve(self) -> KnapsackResult:
        t0 = time.time()

        # All items free at root; bitmask with bits 0..n-1 set
        root_bits  = (1 << self.n) - 1
        root_bound, _ = self._fk(self.banda_cap, 0.0, root_bits)

        # Heap entry: (-bound, counter, rem_banda, rem_buffer, cur_prio,
        #              free_bits, sel_bits)
        # free_bits : bitmask of items not yet fixed (1 = free)
        # sel_bits  : bitmask of items fixed to 1
        counter = 0
        heap: list = [(-root_bound, counter, self.banda_cap, self.buffer_cap,
                        0.0, root_bits, 0)]
        nodes = 0

        while heap:
            if time.time() - t0 > self.time_limit:
                best_bound = -heap[0][0]
                gap = 100.0 * (best_bound - self.incumbent) / max(abs(self.incumbent), 1e-9)
                return KnapsackResult("timeout", self.incumbent,
                                      self.best_x.copy(), root_bound,
                                      best_bound, gap, nodes, time.time() - t0)

            neg_b, _, rem_b, rem_buf, cur_p, free_bits, sel_bits = heapq.heappop(heap)
            bound = -neg_b
            nodes += 1

            if bound <= self.incumbent + 1e-8:
                continue

            # Find break item (most-infeasible = unique fractional var in FK)
            _, break_j = self._fk(rem_b, 0.0, free_bits)

            if break_j is None:
                # FK is integer → try to take all free items respecting buffer
                self._update_incumbent(cur_p, rem_b, rem_buf, free_bits, sel_bits)
                continue

            # Branch on break_j -----------------------------------------------
            remaining_bits = free_bits & ~(1 << break_j)   # clear break item

            # x_{break_j} = 1  (feasibility: both bandwidth and buffer)
            if (rem_b  >= self.banda[break_j]  - 1e-9 and
                    rem_buf >= self.buffer[break_j] - 1e-9):
                new_rem_b   = rem_b   - self.banda[break_j]
                new_rem_buf = rem_buf - self.buffer[break_j]
                new_p       = cur_p   + self.priority[break_j]
                b1, _ = self._fk(new_rem_b, new_p, remaining_bits)
                if b1 > self.incumbent + 1e-8:
                    counter += 1
                    heapq.heappush(heap, (-b1, counter, new_rem_b, new_rem_buf,
                                          new_p, remaining_bits,
                                          sel_bits | (1 << break_j)))

            # x_{break_j} = 0
            b0, _ = self._fk(rem_b, cur_p, remaining_bits)
            if b0 > self.incumbent + 1e-8:
                counter += 1
                heapq.heappush(heap, (-b0, counter, rem_b, rem_buf,
                                      cur_p, remaining_bits, sel_bits))

        elapsed = time.time() - t0
        # Queue exhausted → proven optimal; gap = 0 %
        return KnapsackResult(
            "optimal", self.incumbent, self.best_x.copy(),
            root_bound, self.incumbent, 0.0, nodes, elapsed,
        )


# ------------------------------------------------------------------
# Solution printer (works for both KnapsackResult and BBResult)
# ------------------------------------------------------------------

def print_solution(
    result,
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

    x = result.x if hasattr(result, "x") else None
    obj = result.obj_value

    if x is None or obj is None:
        print("Sem solucao viavel.")
        print("=" * 65)
        return

    print(f"Prioridade opt.: {obj:.2f}")
    # Gap (shown for KnapsackResult; 0 % when optimal)
    if hasattr(result, "root_fk_bound"):
        print(f"Bound FK raiz  : {result.root_fk_bound:.2f}")
        print(f"Gap final      : {result.gap_pct:.1f}%")

    selected = [j for j in range(n) if round(x[j]) == 1]
    total_banda  = sum(banda[j]  for j in selected)
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
    banda    = [rng.randint(10, 50) for _ in range(n)]
    buffer   = [rng.randint(2,  15) for _ in range(n)]
    priority = [rng.randint(10, 100) for _ in range(n)]
    return banda, buffer, priority


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Instance from the assignment ----
    banda_orig    = [30, 20, 45, 15, 35, 25, 40]
    buffer_orig   = [8,   5, 10,  3,  9,  6,  7]
    priority_orig = [50, 30, 70, 20, 60, 40, 65]
    banda_cap  = 100
    buffer_cap = 25

    n_orig = len(banda_orig)
    order = sorted(range(n_orig),
                   key=lambda j: priority_orig[j] / banda_orig[j], reverse=True)

    fk_root = fractional_knapsack_bound(
        banda_orig, priority_orig, float(banda_cap),
        [order[k] for k in range(n_orig)],
    )
    greedy_val, x_greedy = greedy_integer_solution(
        banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap
    )

    print("=" * 65)
    print("PROBLEMA 3 - Instancia do enunciado (7 fluxos)")
    print(f"Banda cap={banda_cap} Mbps | Buffer cap={buffer_cap} MB")
    print()
    print(f"  Ordem greedy (razao p/b desc.): {['F'+str(j+1) for j in order]}")
    print(f"  Bound FK raiz (otimista):        {fk_root:.4f}")
    print(f"  Incumbente greedy inteiro:       {greedy_val:.2f}  "
          f"({[f'F{j+1}' for j in range(n_orig) if x_greedy[j]>0.5]})")
    print(f"  Gap inicial (raiz):  "
          f"{100*(fk_root - greedy_val)/max(abs(greedy_val),1e-9):.1f}%")
    print("=" * 65)

    # ---- B&B com bound FK ----
    print("\n--- B&B (bound mochila fracionaria, best-first) ---")
    bnb = KnapsackBnB(banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap,
                      initial_incumbent=greedy_val, initial_x=x_greedy)
    r_bb = bnb.solve()
    print_solution(r_bb, banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap,
                   "B&B | Instancia original")

    # ---- B&C com cover cuts (LP relaxation + cover cuts) ----
    model_orig = build_model(banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap)
    lp_orig    = solve_relaxation(model_orig)
    lp_bound_orig = lp_orig.obj_value if lp_orig.status == "optimal" else None

    print("\n--- B&C (cover cuts, warm-start greedy) ---")
    bc = BranchAndCut(model_orig, strategy="best_first", branching="most_infeasible",
                      cut_types=["cover"],
                      initial_incumbent=greedy_val, initial_x=x_greedy)
    r_bc = bc.solve()
    print_solution(r_bc, banda_orig, buffer_orig, priority_orig, banda_cap, buffer_cap,
                   "B&C | Instancia original")

    # ---- Random instances ----
    summary: list[tuple] = []
    summary.append(("Original (7 fl.)", "B&B (FK)",  r_bb.obj_value,
                    r_bb.root_fk_bound, r_bb.gap_pct, r_bb.nodes_explored, r_bb.elapsed))
    summary.append(("Original (7 fl.)", "B&C (cov)", r_bc.obj_value,
                    lp_bound_orig, 0.0 if r_bc.status == "optimal" else None,
                    r_bc.nodes_explored, r_bc.elapsed))

    for n_flows, seed in [(10, 1), (20, 2), (50, 3)]:
        banda, buffer, priority = random_instance(n_flows, seed=seed)
        scale   = n_flows / 7
        bc_cap  = int(banda_cap  * scale * 0.6)
        buf_cap = int(buffer_cap * scale * 0.6)

        label = f"Aleat. ({n_flows} fl.)"
        print(f"\n{'=' * 65}")
        print(f"PROBLEMA 3 - {label}")
        print(f"Banda cap={bc_cap} Mbps | Buffer cap={buf_cap} MB")
        print("=" * 65)

        g_val, x_g = greedy_integer_solution(banda, buffer, priority, bc_cap, buf_cap)

        print("\n--- B&B (bound FK) ---")
        bnb_r = KnapsackBnB(banda, buffer, priority, bc_cap, buf_cap,
                             initial_incumbent=g_val, initial_x=x_g,
                             time_limit=60.0)
        r_bb_r = bnb_r.solve()
        print_solution(r_bb_r, banda, buffer, priority, bc_cap, buf_cap,
                       f"B&B | {label}")

        model_r   = build_model(banda, buffer, priority, bc_cap, buf_cap)
        lp_r      = solve_relaxation(model_r)
        lp_bound_r = lp_r.obj_value if lp_r.status == "optimal" else None

        print("\n--- B&C (cover cuts) ---")
        bc_r = BranchAndCut(model_r, strategy="best_first", branching="most_infeasible",
                             cut_types=["cover"],
                             initial_incumbent=g_val if g_val > 0 else None,
                             initial_x=x_g if g_val > 0 else None)
        r_bc_r = bc_r.solve()
        print_solution(r_bc_r, banda, buffer, priority, bc_cap, buf_cap,
                       f"B&C | {label}")

        summary.append((label, "B&B (FK)",  r_bb_r.obj_value,
                        r_bb_r.root_fk_bound, r_bb_r.gap_pct,
                        r_bb_r.nodes_explored, r_bb_r.elapsed))
        summary.append((label, "B&C (cov)", r_bc_r.obj_value,
                        lp_bound_r,
                        0.0 if r_bc_r.status == "optimal" else None,
                        r_bc_r.nodes_explored, r_bc_r.elapsed))

    # ---- Summary table ----
    print("\n" + "=" * 95)
    print("RESUMO DAS INSTANCIAS")
    print("=" * 95)
    hdr = (f"{'Instancia':<18} {'Algoritmo':<12} {'Obj':>8} {'Melhor bound':>13} "
           f"{'Gap (%)':>8} {'Nos':>7} {'Tempo (s)':>10}")
    print(hdr)
    print("-" * 95)
    for inst, algo, obj, bound, gap, nos, t in summary:
        obj_s   = f"{obj:.2f}"   if obj   is not None else "N/A"
        bnd_s   = f"{bound:.2f}" if bound is not None else "N/A"
        gap_s   = f"{gap:.1f}%"  if gap   is not None else "N/A"
        print(f"{inst:<18} {algo:<12} {obj_s:>8} {bnd_s:>13} {gap_s:>8} {nos:>7} {t:>10.4f}")
    print()
    print("  Nota: 'Melhor bound' = bound FK na raiz (B&B) ou LP na raiz (B&C).")
    print("  Gap (%) = 100*(melhor_bound - obj)/|obj|.")
    print("  Gap = 0.0% indica que o algoritmo provou otimalidade.")
