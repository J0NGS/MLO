"""
Problema 7 - Roteamento de Tecnicos (TSP) [PIB]

Um tecnico deve visitar 6 localidades partindo do deposito e
retornando a ele, minimizando o tempo total de deslocamento.

Formulacao PIB:
  DFJ (Branch-and-Cut):
    min  sum_{i!=j} d_{ij} * x_{ij}
    s.t.
    (1)  sum_j x_{ij} = 1   para cada i
    (2)  sum_i x_{ij} = 1   para cada j
    (3)  sum_{i,j in S} x_{ij} <= |S|-1   (SECs lazy via B&C)
         x_{ij} in {0,1},  x_{ii} = 0

  MTZ (Branch-and-Bound):
    Mesmas restricoes de grau + restricoes MTZ:
    u_i - u_j + n*x_{ij} <= n-1   para i,j in {1..n-1}, i!=j
    u_i in {1,..,n-1} (inteiro)

Indexacao DFJ: idx(i,j) = i*n + j  (n*n variaveis)
Indexacao MTZ: n*n variaveis x + (n-1) variaveis u
    idx_u(i) = n*n + (i-1)   para i in {1..n-1}

Instancia (6 cidades, tempos em minutos):
    Deposito: E0 | Clientes: E1, E2, E3, E4, E5

           E0   E1   E2   E3   E4   E5
    E0 [  0,  10,  15,  20,  18,  25 ]
    E1 [ 10,   0,  12,  18,  14,  22 ]
    E2 [ 15,  12,   0,   8,  16,  19 ]
    E3 [ 20,  18,   8,   0,  11,  14 ]
    E4 [ 18,  14,  16,  11,   0,   9 ]
    E5 [ 25,  22,  19,  14,   9,   0 ]

Roteiro otimo: E0->E2->E3->E5->E4->E1->E0, tempo = 70 min.
Heuristica NN: E0->E1->E2->E3->E4->E5->E0, tempo = 75 min.

Analise de sensibilidade (d[E0][E2] em [5..25]):
    Roteiro com E0-E2: custo = d[E0-E2] + 55
    Roteiro sem E0-E2: custo = 71 (E0->E1->E2->E3->E5->E4->E0)
    Transicao em d[E0-E2] = 16:
        d[E0-E2] <= 16 -> usa E0-E2, custo = d + 55
        d[E0-E2] >= 17 -> evita E0-E2, custo = 71
"""
from __future__ import annotations
import sys, os
from itertools import combinations
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult
from core.node import Node


# ------------------------------------------------------------------
# Default instance (6 cities, symmetric distances in minutes)
# ------------------------------------------------------------------

N = 6
CITY_NAMES = [f"E{i}" for i in range(N)]

DEFAULT_DIST: list[list[float]] = [
    [ 0, 10, 15, 20, 18, 25],  # E0
    [10,  0, 12, 18, 14, 22],  # E1
    [15, 12,  0,  8, 16, 19],  # E2
    [20, 18,  8,  0, 11, 14],  # E3
    [18, 14, 16, 11,  0,  9],  # E4
    [25, 22, 19, 14,  9,  0],  # E5
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def idx(i: int, j: int, n: int) -> int:
    """Flat variable index for x_{ij}."""
    return i * n + j


def tour_cost(tour: list[int], dist: list[list[float]]) -> float:
    n = len(tour)
    return sum(dist[tour[k]][tour[(k + 1) % n]] for k in range(n))


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def build_model(
    dist: list[list[float]],
    n: int | None = None,
    include_secs: bool = True,
) -> MIPModel:
    """
    Build TSP MIPModel.

    include_secs: if True, add all subtour elimination constraints (SECs)
                  for subsets of size 2..n-1. Suitable for B&B.
                  If False, only degree constraints are included; SECs are
                  added lazily by TSPBranchAndCut. Suitable for B&C.
    """
    if n is None:
        n = len(dist)
    n_vars = n * n

    # Objective: sum_{i!=j} d_{ij} * x_{ij}
    c = [0.0] * n_vars
    for i in range(n):
        for j in range(n):
            if i != j:
                c[idx(i, j, n)] = float(dist[i][j])

    # Bounds: diagonal fixed to 0, off-diagonal binary [0,1]
    bounds = [(0.0, 1.0)] * n_vars
    for i in range(n):
        bounds[idx(i, i, n)] = (0.0, 0.0)

    # Integrality: 0 for diagonal (continuous, fixed by bounds), 2 for off-diagonal (binary)
    integrality = [0] * n_vars
    for i in range(n):
        for j in range(n):
            if i != j:
                integrality[idx(i, j, n)] = 2

    # Degree constraints (equality)
    A_eq, b_eq = [], []
    for i in range(n):
        # Out-degree: sum_j x_{ij} = 1
        row = [0.0] * n_vars
        for j in range(n):
            row[idx(i, j, n)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    for j in range(n):
        # In-degree: sum_i x_{ij} = 1
        row = [0.0] * n_vars
        for i in range(n):
            row[idx(i, j, n)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    # Subtour elimination constraints (SEC)
    A_ub: list[list[float]] = []
    b_ub: list[float] = []
    if include_secs:
        cities = list(range(n))
        for size in range(2, n):             # subsets of size 2..n-1
            for S in combinations(cities, size):
                row = [0.0] * n_vars
                for i in S:
                    for j in S:
                        if i != j:
                            row[idx(i, j, n)] = 1.0
                A_ub.append(row)
                b_ub.append(float(len(S) - 1))

    var_names = [f"x{i}{j}" for i in range(n) for j in range(n)]

    return MIPModel(
        c=c, A_eq=A_eq, b_eq=b_eq,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        bounds=bounds, integrality=integrality,
        sense="min",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# MTZ model builder (for B&B)
# ------------------------------------------------------------------

def build_model_mtz(
    dist: list[list[float]],
    n: int | None = None,
) -> MIPModel:
    """
    Build TSP MIPModel with Miller-Tucker-Zemlin (MTZ) subtour constraints.

    Variables: n*n arc variables x_{ij} + (n-1) position variables u_i (i=1..n-1).
    idx_u(i) = n*n + (i-1)  for i in {1..n-1}.
    MTZ: u_i - u_j + n*x_{ij} <= n-1  for i,j in {1..n-1}, i!=j.
    u_i bounds: [1, n-1], integrality=1 (general integer).
    """
    if n is None:
        n = len(dist)
    n_vars = n * n + (n - 1)  # x vars + u vars

    def idx_u(i: int) -> int:
        return n * n + (i - 1)

    # Objective over arc variables only
    c = [0.0] * n_vars
    for i in range(n):
        for j in range(n):
            if i != j:
                c[idx(i, j, n)] = float(dist[i][j])

    # Bounds
    bounds = [(0.0, 1.0)] * (n * n) + [(1.0, float(n - 1))] * (n - 1)
    for i in range(n):
        bounds[idx(i, i, n)] = (0.0, 0.0)

    # Integrality: arc vars binary, u vars general integer
    integrality = [0] * n_vars
    for i in range(n):
        for j in range(n):
            if i != j:
                integrality[idx(i, j, n)] = 2
    for i in range(1, n):
        integrality[idx_u(i)] = 1

    # Degree constraints (equality): same as DFJ
    A_eq, b_eq = [], []
    for i in range(n):
        row = [0.0] * n_vars
        for j in range(n):
            row[idx(i, j, n)] = 1.0
        A_eq.append(row); b_eq.append(1.0)
    for j in range(n):
        row = [0.0] * n_vars
        for i in range(n):
            row[idx(i, j, n)] = 1.0
        A_eq.append(row); b_eq.append(1.0)

    # MTZ constraints: u_i - u_j + n*x_{ij} <= n-1  for i,j in {1..n-1}, i!=j
    A_ub: list[list[float]] = []
    b_ub: list[float] = []
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            row = [0.0] * n_vars
            row[idx_u(i)] = 1.0
            row[idx_u(j)] = -1.0
            row[idx(i, j, n)] = float(n)
            A_ub.append(row)
            b_ub.append(float(n - 1))

    var_names = [f"x{i}{j}" for i in range(n) for j in range(n)]
    var_names += [f"u{i}" for i in range(1, n)]

    return MIPModel(
        c=c, A_eq=A_eq, b_eq=b_eq,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        bounds=bounds, integrality=integrality,
        sense="min",
        var_names=var_names,
    )


# ------------------------------------------------------------------
# Nearest-Neighbor heuristic (warm start)
# ------------------------------------------------------------------

def nearest_neighbor_tour(
    dist: list[list[float]],
    n: int | None = None,
    start: int = 0,
) -> tuple[list[int], float]:
    """Greedy nearest-neighbor tour. Returns (tour, cost)."""
    if n is None:
        n = len(dist)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start

    while unvisited:
        nearest = min(unvisited, key=lambda j: dist[cur][j])
        tour.append(nearest)
        unvisited.remove(nearest)
        cur = nearest

    return tour, tour_cost(tour, dist)


def tour_to_x_vector(tour: list[int], n: int) -> np.ndarray:
    """Encode a tour as the flat x vector."""
    x = np.zeros(n * n)
    m = len(tour)
    for k in range(m):
        i = tour[k]
        j = tour[(k + 1) % m]
        x[idx(i, j, n)] = 1.0
    return x


# ------------------------------------------------------------------
# Solution decoder
# ------------------------------------------------------------------

def decode_tour(result: BBResult, n: int) -> list[int] | None:
    """Reconstruct ordered tour from solution vector."""
    if result.x is None:
        return None
    nxt: dict[int, int] = {}
    for i in range(n):
        for j in range(n):
            if i != j and round(result.x[idx(i, j, n)]) == 1:
                nxt[i] = j
    if len(nxt) != n:
        return None
    tour = [0]
    cur = 0
    for _ in range(n - 1):
        cur = nxt.get(cur, -1)
        if cur < 0:
            return None
        tour.append(cur)
    return tour


# ------------------------------------------------------------------
# TSPBranchAndCut: adds lazy subtour cuts in the B&C loop
# ------------------------------------------------------------------

class TSPBranchAndCut(BranchAndCut):
    """
    B&C for TSP com dois modos:

    mtz_mode=False (padrao): parte do modelo de grau sem SECs; adiciona
        cortes DFJ (subtour) e Gomory lazily. Precisa de _is_valid_tour
        porque o modelo base nao tem garantia de tour valido em solucoes
        inteiras.

    mtz_mode=True: parte do modelo MTZ, que ja elimina subtours em
        solucoes inteiras pelas restricoes u_i - u_j + n*x_{ij} <= n-1.
        Adiciona cortes Gomory (e opcionalmente DFJ) sobre a relaxacao
        MTZ. Nao precisa de _is_valid_tour.
    """

    def __init__(self, n_cities: int, *args, mtz_mode: bool = False, **kwargs):
        self.n_cities = n_cities
        self.mtz_mode = mtz_mode
        super().__init__(*args, **kwargs)

    # --- Integrality check ---

    def _is_integer(self, x: np.ndarray, tol: float = 1e-5) -> bool:
        """MTZ mode: integridade suficiente (MTZ garante tour valido).
        DFJ mode: rejeita solucoes inteiras com subtours."""
        if not super()._is_integer(x, tol):
            return False
        if self.mtz_mode:
            return True  # restricoes MTZ garantem tour valido para solucoes inteiras
        return self._is_valid_tour(x)

    def _is_valid_tour(self, x: np.ndarray) -> bool:
        """Return True iff x encodes a single cycle visiting all n cities."""
        n = self.n_cities
        nxt: dict[int, int] = {}
        for i in range(n):
            for j in range(n):
                if i != j and x[idx(i, j, n)] > 0.5:
                    nxt[i] = j
        if len(nxt) != n:
            return False
        visited: set[int] = set()
        cur = 0
        while cur not in visited:
            visited.add(cur)
            cur = nxt.get(cur, -1)
            if cur < 0:
                return False
        return cur == 0 and len(visited) == n

    # --- Cut generation ---

    def _generate_cuts(
        self,
        x: np.ndarray,
        node: Node,
        lp_result=None,
        cut_details: list | None = None,
    ) -> list[tuple[str, np.ndarray, float]]:
        cuts = super()._generate_cuts(x, node, lp_result, cut_details=cut_details)
        if "subtour" in self.cut_types:
            cuts += self._subtour_cuts(x)
        return cuts

    def _subtour_cuts(self, x: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        """
        Detecta subtours via componentes conexas do grafo suporte
        (arcos com x_{ij} > 0.5). Para cada componente propria S,
        adiciona a SEC violada: sum_{i,j in S, i!=j} x_{ij} <= |S|-1.

        O vetor cut_lhs tem dimensao len(x) para ser compativel tanto
        com o modelo DFJ (n*n variaveis) quanto com o MTZ (n*n + (n-1)).
        """
        n = self.n_cities
        n_vars = len(x)  # compativel com DFJ e MTZ

        succ: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and x[idx(i, j, n)] > 0.5:
                    succ[i].append(j)

        visited = [False] * n
        components: list[list[int]] = []
        for start in range(n):
            if not visited[start]:
                comp: list[int] = []
                stack = [start]
                while stack:
                    cur = stack.pop()
                    if visited[cur]:
                        continue
                    visited[cur] = True
                    comp.append(cur)
                    for nb in succ[cur]:
                        if not visited[nb]:
                            stack.append(nb)
                    for i in range(n):
                        if not visited[i] and cur in succ[i]:
                            stack.append(i)
                components.append(comp)

        if len(components) <= 1:
            return []

        cuts: list[tuple[str, np.ndarray, float]] = []
        for comp in components:
            if len(comp) < n:
                S = comp
                cut_lhs = np.zeros(n_vars)  # zeros nas vars u para modelo MTZ
                for i in S:
                    for j in S:
                        if i != j:
                            cut_lhs[idx(i, j, n)] = 1.0
                cut_rhs = float(len(S) - 1)
                if cut_lhs @ x > cut_rhs + 1e-6:
                    cuts.append(("subtour", cut_lhs, cut_rhs))
        return cuts


# ------------------------------------------------------------------
# Printer
# ------------------------------------------------------------------

def print_solution(
    result: BBResult,
    dist: list[list[float]],
    n: int,
    label: str = "",
    city_names: list[str] | None = None,
):
    if city_names is None:
        city_names = [f"C{i}" for i in range(n)]

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

    print(f"Custo otimo     : {result.obj_value:.1f} min")
    tour = decode_tour(result, n)
    if tour:
        route = " -> ".join(city_names[c] for c in tour) + f" -> {city_names[tour[0]]}"
        print(f"Roteiro         : {route}")
        print()
        print("  Trecho          Tempo")
        print("  " + "-" * 25)
        total = 0.0
        for k in range(n):
            i, j = tour[k], tour[(k + 1) % n]
            t = dist[i][j]
            total += t
            print(f"  {city_names[i]} -> {city_names[j]}   {t:>5.0f} min")
        print("  " + "-" * 25)
        print(f"  Total           {total:>5.0f} min")
    print("=" * 65)


# ------------------------------------------------------------------
# Sensitivity analysis: vary d[E0][E2] (symmetric)
# ------------------------------------------------------------------

def sensitivity_analysis(
    d02_range: list[float],
    dist_base: list[list[float]] = DEFAULT_DIST,
    n: int = N,
) -> list[tuple[float, float | None, list[int] | None]]:
    """
    Vary d[0][2] = d[2][0] and solve the TSP with MTZ model (B&B).
    Returns list of (d02, optimal_cost, tour).
    """
    results = []
    for d02 in d02_range:
        dist = [list(row) for row in dist_base]
        dist[0][2] = d02
        dist[2][0] = d02
        model = build_model_mtz(dist, n)
        nn_tour, nn_cost = nearest_neighbor_tour(dist, n)
        x_nn_full = np.zeros(n * n + (n - 1))
        x_nn_full[:n * n] = tour_to_x_vector(nn_tour, n)
        solver = BranchAndBound(model, strategy="best_first",
                                branching="most_infeasible",
                                initial_incumbent=nn_cost, initial_x=x_nn_full,
                                verbose=False)
        r = solver.solve()
        tour = decode_tour(r, n) if r.x is not None else None
        results.append((d02, r.obj_value, tour))
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    cn = CITY_NAMES

    print("=" * 65)
    print("PROBLEMA 7 - Roteamento de Tecnicos (TSP - MTZ/DFJ)")
    print("=" * 65)
    print(f"\n{N} cidades | Matriz de distancias (min):\n")
    header = "        " + "  ".join(f"{cn[j]:>4}" for j in range(N))
    print(header)
    for i in range(N):
        row_str = "  ".join(f"{DEFAULT_DIST[i][j]:>4.0f}" for j in range(N))
        print(f"  {cn[i]}:  {row_str}")

    # Nearest-neighbor warm start
    nn_tour, nn_cost = nearest_neighbor_tour(DEFAULT_DIST, N)
    nn_route = " -> ".join(cn[c] for c in nn_tour) + f" -> {cn[nn_tour[0]]}"
    print(f"\nHeuristica NN: {nn_route} | custo = {nn_cost:.0f} min")
    x_nn = tour_to_x_vector(nn_tour, N)

    # B&B with MTZ model
    model_mtz = build_model_mtz(DEFAULT_DIST, N)
    n_mtz_vars = N * N + (N - 1)
    x_nn_mtz = np.zeros(n_mtz_vars)
    x_nn_mtz[:N * N] = x_nn

    print(f"\n(Modelo MTZ: {N*N} variaveis de arco + {N-1} variaveis de posicao u_i)")

    print("\n--- Branch-and-Bound (modelo MTZ) ---")
    bb = BranchAndBound(model_mtz, strategy="best_first", branching="most_infeasible",
                        initial_incumbent=nn_cost, initial_x=x_nn_mtz)
    r_bb = bb.solve()
    print_solution(r_bb, DEFAULT_DIST, N, label="B&B MTZ | Instancia original", city_names=cn)

    # B&C com relaxacao MTZ em cada no (alinhado ao enunciado)
    # Modelo base: MTZ (grau + restricoes u_i - u_j + n*x_ij <= n-1)
    # Cortes: Gomory sobre a relaxacao MTZ + SECs DFJ lazily
    # mtz_mode=True: _is_integer nao precisa verificar tour valido
    #   (as restricoes MTZ garantem isso para solucoes inteiras)
    x_nn_mtz_bc = np.zeros(N * N + (N - 1))
    x_nn_mtz_bc[:N * N] = x_nn

    print(f"\n(Modelo MTZ: relaxacao MTZ em cada no, cortes Gomory + SECs DFJ lazy)")
    print("\n--- Branch-and-Cut (relaxacao MTZ + Gomory + SECs lazy) ---")
    bc = TSPBranchAndCut(N, model_mtz, strategy="best_first",
                         branching="most_infeasible",
                         mtz_mode=True,
                         cut_types=["gomory", "subtour"],
                         initial_incumbent=nn_cost, initial_x=x_nn_mtz_bc)
    r_bc = bc.solve()
    print_solution(r_bc, DEFAULT_DIST, N, label="B&C MTZ | Instancia original", city_names=cn)

    print(f"\nB&B (MTZ) nos={r_bb.nodes_explored} | B&C (MTZ + cortes) nos={r_bc.nodes_explored}")
    if bc._cut_log:
        subtour_cuts = sum(1 for e in bc._cut_log if e["type"] == "subtour")
        gomory_cuts  = sum(1 for e in bc._cut_log if e["type"] == "gomory")
        print(f"Cortes adicionados pelo B&C: {gomory_cuts} Gomory, {subtour_cuts} SEC")

    # ---- Sensitivity: vary d[E0][E2] ----
    print("\n" + "=" * 65)
    print("ANALISE DE SENSIBILIDADE — distancia d[E0][E2]")
    print("=" * 65)
    d02_range = [5.0, 10.0, 15.0, 16.0, 17.0, 20.0, 25.0]
    sens = sensitivity_analysis(d02_range)

    print(f"\n  {'d[E0-E2]':>10} {'Custo':>8}  Roteiro otimo")
    print("  " + "-" * 60)
    prev_uses_02 = None
    for d02, cost, tour in sens:
        if cost is None:
            print(f"  {d02:>10.0f} {'inviavel':>8}")
            continue
        uses_02 = tour is not None and any(
            (tour[k] == 0 and tour[(k+1)%N] == 2) or
            (tour[k] == 2 and tour[(k+1)%N] == 0)
            for k in range(N)
        )
        flag = " (usa E0-E2)" if uses_02 else ""
        change = " <-- mudanca" if prev_uses_02 is not None and uses_02 != prev_uses_02 else ""
        route_str = " -> ".join(f"E{c}" for c in tour) + f" -> E{tour[0]}" if tour else "?"
        print(f"  {d02:>10.0f} {cost:>8.1f}  {route_str}{flag}{change}")
        prev_uses_02 = uses_02

    print("\nInterpretacao:")
    print("  d[E0-E2] <= 16 -> usa arco E0-E2, custo = d[E0-E2] + 55")
    print("  d[E0-E2] =  16 -> empate (custo 71): dois roteiros igualmente otimos")
    print("  d[E0-E2] >= 17 -> evita E0-E2, custo = 71 fixo (E0->E1->E2->E3->E5->E4->E0)")
