"""
Problema 7 - Roteamento de Tecnicos (TSP) [PIB]

Um tecnico deve visitar 5 localidades partindo do deposito e
retornando a ele, minimizando o tempo total de deslocamento.

Formulacao PIB (Traveling Salesman Problem - formulacao DFJ):
    min  sum_{i!=j} d_{ij} * x_{ij}
    s.t.
    (1)  sum_j x_{ij} = 1   para cada i      (sai de cada cidade exatamente uma vez)
    (2)  sum_i x_{ij} = 1   para cada j      (entra em cada cidade exatamente uma vez)
    (3)  sum_{i in S, j in S, i!=j} x_{ij} <= |S|-1   para todo S subconjunto proprio,
                                                        2 <= |S| <= n-1  (anti-subroteiro)
         x_{ij} in {0,1},  x_{ii} = 0

Indexacao: idx(i,j) = i*n + j  (n*n variaveis; diagonal forcada a 0 pelos limites)

Instancia (5 cidades, tempos em minutos):
    Deposito: C0 | Clientes: C1, C2, C3, C4

         C0   C1   C2   C3   C4
    C0 [  0,  15,  20,  30,  10 ]
    C1 [ 15,   0,  35,  25,  20 ]
    C2 [ 20,  35,   0,  15,  40 ]
    C3 [ 30,  25,  15,   0,  20 ]
    C4 [ 10,  20,  40,  20,   0 ]

Roteiro otimo: C0->C4->C1->C3->C2->C0, tempo = 90 min.

LP-relaxacao (apenas restricoes de grau): custo = 75 (subroteiros {C0,C4,C1} + {C2,C3}).
O B&C adiciona SECs ao detectar subroteiros, convergindo para o otimo de 90.

Estrategia de B&C:
  - B&B: modelo completo com todos os SECs incluidos desde o inicio
  - B&C: inicia apenas com restricoes de grau; SECs sao adicionados como
         cortes lazily ao detectar subroteiros na solucao LP/inteira.
         _is_integer() retorna False para solucoes com subroteiros (mesmo
         que todos x_{ij} sejam binarios), forcando a geracao de cortes.

Analise de sensibilidade (d[C0][C2] em [5..30]):
    Melhor roteiro usando arco C0-C2: custo = d[C0-C2] + 70
    Melhor roteiro evitando C0-C2:    custo = 95 (C0->C4->C3->C2->C1->C0)
    Transicao em d[C0-C2] = 25:
        d[C0-C2] <= 25 -> usa C0-C2, custo = d + 70
        d[C0-C2] >  25 -> evita C0-C2, custo = 95 (constante)
"""
from __future__ import annotations
import sys, os
from itertools import combinations
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import MIPModel, BranchAndBound, BranchAndCut, BBResult
from core.node import Node


# ------------------------------------------------------------------
# Default instance (5 cities, symmetric distances in minutes)
# ------------------------------------------------------------------

N = 5
CITY_NAMES = [f"C{i}" for i in range(N)]

DEFAULT_DIST: list[list[float]] = [
    [ 0, 15, 20, 30, 10],
    [15,  0, 35, 25, 20],
    [20, 35,  0, 15, 40],
    [30, 25, 15,  0, 20],
    [10, 20, 40, 20,  0],
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
    B&C for TSP that detects subtours in the LP solution and adds the
    violated SEC as a cut. Starts from the degree-only model and adds
    SECs lazily, mimicking the classic B&C approach for TSP.

    Key override: _is_integer() returns False for integer solutions that still
    contain subtours, so the cut loop runs and adds the missing SECs before
    any subtour-invalid solution is accepted as an incumbent.
    """

    def __init__(self, n_cities: int, *args, **kwargs):
        self.n_cities = n_cities
        super().__init__(*args, **kwargs)

    # --- Integrality check: reject solutions with subtours ---

    def _is_integer(self, x: np.ndarray, tol: float = 1e-5) -> bool:
        """Return True only when x is integer AND forms a valid Hamiltonian tour."""
        if not super()._is_integer(x, tol):
            return False
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
    ) -> list[tuple[np.ndarray, float]]:
        # Standard cuts (Gomory, etc.) from parent
        cuts = super()._generate_cuts(x, node, lp_result)
        # Lazy subtour elimination cuts
        if "subtour" in self.cut_types:
            cuts += self._subtour_cuts(x)
        return cuts

    def _subtour_cuts(self, x: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """
        Detect subtours via connected components of the support graph
        (edges with x_{ij} > 0.5). For each proper component S found,
        add the violated SEC: sum_{i,j in S, i!=j} x_{ij} <= |S|-1.
        """
        n = self.n_cities
        # Support graph: directed arcs with x > 0.5 (near-integer threshold)
        succ: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and x[idx(i, j, n)] > 0.5:
                    succ[i].append(j)

        # Find weakly-connected components (ignore arc direction)
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
            return []  # No subtour detected

        cuts = []
        for comp in components:
            if len(comp) < n:  # Proper subset = subtour
                S = comp
                cut_lhs = np.zeros(n * n)
                for i in S:
                    for j in S:
                        if i != j:
                            cut_lhs[idx(i, j, n)] = 1.0
                cut_rhs = float(len(S) - 1)
                if cut_lhs @ x > cut_rhs + 1e-6:
                    cuts.append((cut_lhs, cut_rhs))
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
# Sensitivity analysis: vary d[C0][C2] (symmetric)
# ------------------------------------------------------------------

def sensitivity_analysis(
    d02_range: list[float],
    dist_base: list[list[float]] = DEFAULT_DIST,
    n: int = N,
) -> list[tuple[float, float | None, list[int] | None]]:
    """
    Vary d[0][2] = d[2][0] and solve the TSP with full SEC model.
    Returns list of (d02, optimal_cost, tour).
    """
    results = []
    for d02 in d02_range:
        dist = [list(row) for row in dist_base]
        dist[0][2] = d02
        dist[2][0] = d02
        model = build_model(dist, n, include_secs=True)
        nn_tour, nn_cost = nearest_neighbor_tour(dist, n)
        x_nn = tour_to_x_vector(nn_tour, n)
        solver = BranchAndBound(model, strategy="best_first",
                                branching="most_infeasible",
                                initial_incumbent=nn_cost, initial_x=x_nn,
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
    print("PROBLEMA 7 - Roteamento de Tecnicos (TSP - DFJ)")
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

    # B&B with full model (all SECs included)
    model_full = build_model(DEFAULT_DIST, N, include_secs=True)
    print(f"\n(Modelo completo: {len(model_full.b_ub)} SECs incluidas desde o inicio)")

    print("\n--- Branch-and-Bound (modelo com todos os SECs) ---")
    bb = BranchAndBound(model_full, strategy="best_first", branching="most_infeasible",
                        initial_incumbent=nn_cost, initial_x=x_nn)
    r_bb = bb.solve()
    print_solution(r_bb, DEFAULT_DIST, N, label="B&B | Instancia original", city_names=cn)

    # B&C with lazy SECs
    model_degree = build_model(DEFAULT_DIST, N, include_secs=False)
    print(f"\n(Modelo lazy: apenas restricoes de grau, SECs adicionadas como cortes)")

    print("\n--- Branch-and-Cut (SECs lazily + Gomory) ---")
    bc = TSPBranchAndCut(N, model_degree, strategy="best_first",
                         branching="most_infeasible",
                         cut_types=["subtour", "gomory"],
                         initial_incumbent=nn_cost, initial_x=x_nn)
    r_bc = bc.solve()
    print_solution(r_bc, DEFAULT_DIST, N, label="B&C lazy | Instancia original", city_names=cn)

    print(f"\nB&B nos={r_bb.nodes_explored} | B&C nos={r_bc.nodes_explored}")

    # ---- Sensitivity: vary d[C0][C2] ----
    print("\n" + "=" * 65)
    print("ANALISE DE SENSIBILIDADE — distancia d[C0][C2]")
    print("=" * 65)
    d02_range = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    sens = sensitivity_analysis(d02_range)

    print(f"\n  {'d[C0-C2]':>10} {'Custo':>8}  Roteiro otimo")
    print("  " + "-" * 60)
    prev_uses_02 = None
    for d02, cost, tour in sens:
        if cost is None:
            print(f"  {d02:>10.0f} {'inviavel':>8}")
            continue
        route = " ".join(f"C{c}" for c in tour) + f" C{tour[0]}" if tour else "?"
        # Check if direct C0-C2 or C2-C0 arc is used
        uses_02 = tour is not None and any(
            (tour[k] == 0 and tour[(k+1)%N] == 2) or
            (tour[k] == 2 and tour[(k+1)%N] == 0)
            for k in range(N)
        )
        flag = " (usa C0-C2)" if uses_02 else ""
        change = " <-- mudanca" if prev_uses_02 is not None and uses_02 != prev_uses_02 else ""
        route_str = " -> ".join(f"C{c}" for c in tour) + f" -> C{tour[0]}" if tour else "?"
        print(f"  {d02:>10.0f} {cost:>8.1f}  {route_str}{flag}{change}")
        prev_uses_02 = uses_02

    print("\nInterpretacao:")
    print("  d[C0-C2] <= 25 -> usa arco C0-C2, custo = d[C0-C2] + 70")
    print("  d[C0-C2] =  25 -> empate (custo 95): dois roteiros igualmente otimos")
    print("  d[C0-C2] >  25 -> evita C0-C2, custo = 95 fixo (C0->C4->C3->C2->C1->C0)")
