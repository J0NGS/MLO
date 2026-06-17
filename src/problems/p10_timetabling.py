"""
Problema 10 - Timetabling / Grade de Horarios

Formulacao (MIP):
  x_{ijk} in {0,1}: disciplina i alocada a sala j no horario k

  max  sum_{i,j,k} sat_{ijk} * x_{ijk}

  s.t.
    (1) sum_{j,k} x_{ijk} = 1                     para cada i    [alocacao unica]
    (2) sum_i    x_{ijk} <= 1                      para cada j,k  [conflito de sala]
    (3) x_{ijk} = 0 se cap_j < enroll_i ou indisponivel          [capacidade/disp]
    (5) sum_j x_{D2,j,k1} + sum_j x_{D4,j,k2} <= 1  para k1>=k2 [precedencia D2<D4]
    (6) sum_j x_{D1,j,k} + sum_j x_{D3,j,k} <= 1   para cada k  [anti-conflito D1-D3]

Cortes:
  - Clique cuts: grafo de conflito entre disciplinas; para clique C e horario k:
      sum_{i in C} sum_j x_{ijk} <= 1
  - Cortes de Gomory: a partir do tableau simplex

B&C: estrategia depth-first, TimetablingBranchAndCut
Comparacao: OR-Tools CP-SAT (se instalado)
"""
from __future__ import annotations
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.model import MIPModel
from core.branch_cut import BranchAndCut
from core.cuts import gomory_cuts

try:
    from ortools.sat.python import cp_model as _cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


# ======================================================================
# Estrutura de dados da instancia
# ======================================================================

class TimetablingInstance:
    """
    Instancia do problema de grade de horarios.

    sat[i][j][k] = satisfacao de alocar disciplina i na sala j no horario k.
    None indica alocacao inviavel (capacidade insuficiente ou indisponibilidade).
    """

    def __init__(
        self,
        disc_names: list[str],
        enrollments: list[int],
        room_names: list[str],
        capacities: list[int],
        slot_names: list[str],
        sat: list[list[list[float | None]]],
        precedence: list[tuple[int, int]] | None = None,
        anticonflict: list[tuple[int, int]] | None = None,
    ):
        self.disc_names  = disc_names
        self.enrollments = enrollments
        self.room_names  = room_names
        self.capacities  = capacities
        self.slot_names  = slot_names
        self.sat         = sat
        self.precedence  = precedence  or []
        self.anticonflict= anticonflict or []

        self.nd = len(disc_names)
        self.nr = len(room_names)
        self.nt = len(slot_names)

    def idx(self, i: int, j: int, k: int) -> int:
        return i * (self.nr * self.nt) + j * self.nt + k

    def n_vars(self) -> int:
        return self.nd * self.nr * self.nt

    def is_feasible(self, i: int, j: int, k: int) -> bool:
        return self.sat[i][j][k] is not None


# ======================================================================
# Instancias
# ======================================================================

def make_small_instance() -> TimetablingInstance:
    """4 disciplinas, 3 salas, 4 horarios."""
    disc_names  = ["D1", "D2", "D3", "D4"]
    enrollments = [28,   45,   22,   40]
    room_names  = ["R1", "R2", "R3"]
    capacities  = [30,   55,   25]
    slot_names  = ["T1", "T2", "T3", "T4"]

    # preferencia de horario por disciplina (0..1)
    time_pref = [
        [0.9, 0.7, 0.5, 0.2],  # D1
        [0.8, 0.9, 0.6, 0.3],  # D2
        [0.4, 0.7, 0.9, 0.8],  # D3
        [0.2, 0.5, 0.8, 0.9],  # D4
    ]
    # disponibilidade de sala por disciplina
    room_avail = [
        [True,  True,  False],  # D1: R1(30>=28), R2(55>=28)
        [False, True,  False],  # D2: apenas R2(55>=45)
        [True,  False, True ],  # D3: R1(30>=22), R3(25>=22)
        [False, True,  False],  # D4: apenas R2(55>=40)
    ]

    nd, nr, nt = 4, 3, 4
    sat: list[list[list[float | None]]] = [
        [[None] * nt for _ in range(nr)] for _ in range(nd)
    ]
    for i in range(nd):
        for j in range(nr):
            if not room_avail[i][j]:
                continue
            if capacities[j] < enrollments[i]:
                continue
            fill = enrollments[i] / capacities[j]  # taxa de ocupacao (maior = melhor fit)
            for k in range(nt):
                sat[i][j][k] = round(0.5 * time_pref[i][k] + 0.5 * fill, 4)

    return TimetablingInstance(
        disc_names, enrollments, room_names, capacities, slot_names, sat,
        precedence  = [(1, 3)],  # D2 deve preceder D4
        anticonflict= [(0, 2)],  # D1 nao compartilha horario com D3
    )


def make_large_instance() -> TimetablingInstance:
    """8 disciplinas, 5 salas, 6 horarios."""
    disc_names  = ["CS1", "CS2", "MATH", "PHYS", "ENG", "BIO", "CHEM", "STAT"]
    enrollments = [35,    50,    60,     45,     30,    25,    55,     20]
    room_names  = ["R1", "R2", "R3", "R4", "R5"]
    capacities  = [40,   60,   70,   65,   80]
    slot_names  = ["T1", "T2", "T3", "T4", "T5", "T6"]

    time_pref = [
        [0.9, 0.7, 0.5, 0.3, 0.1, 0.0],  # CS1
        [0.8, 0.9, 0.6, 0.4, 0.2, 0.1],  # CS2
        [0.7, 0.8, 0.9, 0.5, 0.3, 0.2],  # MATH
        [0.5, 0.6, 0.8, 0.9, 0.7, 0.4],  # PHYS
        [0.3, 0.5, 0.7, 0.8, 0.9, 0.6],  # ENG
        [0.2, 0.4, 0.6, 0.7, 0.8, 0.9],  # BIO
        [0.6, 0.7, 0.5, 0.4, 0.3, 0.2],  # CHEM
        [0.4, 0.3, 0.5, 0.6, 0.7, 0.8],  # STAT
    ]
    # disponibilidade de sala (linha=disc, col=sala)
    room_avail = [
        #R1     R2     R3     R4     R5
        [True,  True,  False, False, False],  # CS1  (35)  -> R1(40), R2(60)
        [False, True,  False, True,  False],  # CS2  (50)  -> R2(60), R4(65)
        [False, False, True,  True,  True ],  # MATH (60)  -> R3(70), R4(65), R5(80)
        [False, True,  True,  False, False],  # PHYS (45)  -> R2(60), R3(70)
        [True,  True,  False, False, False],  # ENG  (30)  -> R1(40), R2(60)
        [True,  False, True,  False, False],  # BIO  (25)  -> R1(40), R3(70)
        [False, False, False, True,  True ],  # CHEM (55)  -> R4(65), R5(80)
        [True,  True,  True,  False, False],  # STAT (20)  -> R1(40), R2(60), R3(70)
    ]

    nd, nr, nt = 8, 5, 6
    sat: list[list[list[float | None]]] = [
        [[None] * nt for _ in range(nr)] for _ in range(nd)
    ]
    for i in range(nd):
        for j in range(nr):
            if not room_avail[i][j]:
                continue
            if capacities[j] < enrollments[i]:
                continue
            fill = enrollments[i] / capacities[j]
            for k in range(nt):
                sat[i][j][k] = round(0.5 * time_pref[i][k] + 0.5 * fill, 4)

    return TimetablingInstance(
        disc_names, enrollments, room_names, capacities, slot_names, sat,
        precedence  = [(1, 3)],  # CS2 deve preceder PHYS
        anticonflict= [(2, 0)],  # MATH nao compartilha horario com CS1
    )


# ======================================================================
# Construcao do MIPModel
# ======================================================================

def build_model(inst: TimetablingInstance) -> MIPModel:
    nd, nr, nt = inst.nd, inst.nr, inst.nt
    n = inst.n_vars()

    # --- objetivo: maximizar satisfacao ---
    c = np.zeros(n)
    for i in range(nd):
        for j in range(nr):
            for k in range(nt):
                s = inst.sat[i][j][k]
                if s is not None:
                    c[inst.idx(i, j, k)] = s

    # --- bounds: ub=0 para alocacoes infeasiveis ---
    bounds: list[tuple[float, float]] = []
    for i in range(nd):
        for j in range(nr):
            for k in range(nt):
                ub = 0.0 if inst.sat[i][j][k] is None else 1.0
                bounds.append((0.0, ub))

    # --- integrality: todas binarias ---
    integrality = np.ones(n, dtype=int)

    # --- nomes das variaveis ---
    var_names = []
    for i in range(nd):
        for j in range(nr):
            for k in range(nt):
                var_names.append(
                    f"x_{inst.disc_names[i]}_{inst.room_names[j]}_{inst.slot_names[k]}"
                )

    # ---- A_eq: restricao (1) alocacao unica ----
    # sum_{j,k} x_{ijk} = 1  para cada i
    A_eq = np.zeros((nd, n))
    for i in range(nd):
        for j in range(nr):
            for k in range(nt):
                A_eq[i, inst.idx(i, j, k)] = 1.0
    b_eq = np.ones(nd)

    # ---- A_ub: restricoes de desigualdade ----
    ub_rows: list[tuple[np.ndarray, float]] = []

    # (2) conflito de sala: sum_i x_{ijk} <= 1  para cada (j, k)
    for j in range(nr):
        for k in range(nt):
            row = np.zeros(n)
            for i in range(nd):
                row[inst.idx(i, j, k)] = 1.0
            ub_rows.append((row, 1.0))

    # (5) precedencia: i_from deve ser alocado num horario ESTRITAMENTE anterior a i_to.
    #     Par invalido: i_from em k1, i_to em k2, com k1 >= k2 (i_from nao antes de i_to).
    #     Restricao: sum_j x_{from,j,k1} + sum_j x_{to,j,k2} <= 1  para cada k1 >= k2.
    for (i_from, i_to) in inst.precedence:
        for k1 in range(nt):            # horario de i_from
            for k2 in range(k1 + 1):   # horario de i_to: k2 <= k1 => par invalido
                row = np.zeros(n)
                for j in range(nr):
                    row[inst.idx(i_from, j, k1)] += 1.0
                    row[inst.idx(i_to,   j, k2)] += 1.0
                ub_rows.append((row, 1.0))

    # (6) anti-conflito: sum_j x_{i1,j,k} + sum_j x_{i2,j,k} <= 1  para cada k
    for (i1, i2) in inst.anticonflict:
        for k in range(nt):
            row = np.zeros(n)
            for j in range(nr):
                row[inst.idx(i1, j, k)] = 1.0
                row[inst.idx(i2, j, k)] = 1.0
            ub_rows.append((row, 1.0))

    A_ub = np.array([r for r, _ in ub_rows])
    b_ub = np.array([b for _, b in ub_rows])

    return MIPModel(
        c=c, bounds=bounds, integrality=integrality,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        sense="max",
        var_names=var_names,
    )


# ======================================================================
# Grafo de conflito e cliques maximais
# ======================================================================

def build_conflict_graph(inst: TimetablingInstance) -> list[set[int]]:
    """
    adj[i] = disciplinas que conflitam com i (impossivel dividir horario).

    Duas disciplinas conflitam se, em QUALQUER atribuicao de sala, nao podem
    estar no mesmo horario:

      a) Restricao explicita de anti-conflito.
      b) Sala unica compartilhada: ambas possuem exatamente UMA sala disponivel
         e ela e a mesma sala. Qualquer alocacao no mesmo horario implicaria
         conflito de sala, que ja e proibido pela restricao (2).

    Nota: "i tem sala R2 e i' tem salas {R1,R2}" NAO e conflito - i' pode usar
    R1 enquanto i usa R2 no mesmo horario, sem violar nenhuma restricao.
    """
    nd, nr = inst.nd, inst.nr
    adj = [set() for _ in range(nd)]

    # a) anti-conflitos explicitos
    for (i1, i2) in inst.anticonflict:
        adj[i1].add(i2)
        adj[i2].add(i1)

    # b) sala unica compartilhada: room_sets[i] == room_sets[i'] == {j}
    room_sets = []
    for i in range(nd):
        rs = frozenset(
            j for j in range(nr)
            if any(inst.sat[i][j][k] is not None for k in range(inst.nt))
        )
        room_sets.append(rs)

    for i in range(nd):
        if len(room_sets[i]) != 1:
            continue          # i tem mais de uma sala disponivel; sem conflito por sala
        for ip in range(i + 1, nd):
            if len(room_sets[ip]) != 1:
                continue
            if room_sets[i] == room_sets[ip]:  # mesma sala unica
                adj[i].add(ip)
                adj[ip].add(i)

    return adj


def _bron_kerbosch(
    R: set, P: set, X: set,
    adj: list[set[int]],
    cliques: list[list[int]],
) -> None:
    """Bron-Kerbosch com pivo para cliques maximais."""
    if not P and not X:
        if len(R) >= 2:
            cliques.append(sorted(R))
        return
    pivot = max(P | X, key=lambda v: len(adj[v] & P))
    for v in list(P - adj[pivot]):
        _bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques)
        P.remove(v)
        X.add(v)


def find_maximal_cliques(adj: list[set[int]]) -> list[list[int]]:
    """Retorna todas as cliques maximais (tamanho >= 2)."""
    cliques: list[list[int]] = []
    _bron_kerbosch(set(), set(range(len(adj))), set(), adj, cliques)
    return cliques


# ======================================================================
# B&C com cortes de clique
# ======================================================================

class TimetablingBranchAndCut(BranchAndCut):
    """
    B&C para timetabling com cortes de clique adicionais.

    Para cada clique maximal C e horario k:
      sum_{i in C} sum_j x_{ijk} <= 1
    Ativado quando violado pela solucao LP atual.
    """

    def __init__(self, inst: TimetablingInstance, *args, **kwargs):
        self.inst   = inst
        adj         = build_conflict_graph(inst)
        self.cliques= find_maximal_cliques(adj)
        super().__init__(*args, **kwargs)

    def _generate_cuts(self, x, node, lp_result=None, cut_details=None):
        cuts = super()._generate_cuts(x, node, lp_result, cut_details=cut_details)
        if "clique" in self.cut_types:
            cuts += self._clique_cuts(x, cut_details=cut_details)
        return cuts

    def _clique_cuts(
        self, x: np.ndarray, tol: float = 1e-6,
        cut_details: list | None = None,
    ) -> list[tuple[str, np.ndarray, float]]:
        inst        = self.inst
        nd, nr, nt  = inst.nd, inst.nr, inst.nt
        n           = inst.n_vars()
        cuts        = []

        # y[i][k] = sum_j x_{ijk}: fracao do horario k usada pela disciplina i
        y = np.zeros((nd, nt))
        for i in range(nd):
            for j in range(nr):
                for k in range(nt):
                    y[i, k] += x[inst.idx(i, j, k)]

        for clique in self.cliques:
            for k in range(nt):
                lhs_val = sum(y[i, k] for i in clique)
                if lhs_val > 1.0 + tol:
                    row = np.zeros(n)
                    for i in clique:
                        for j in range(nr):
                            row[inst.idx(i, j, k)] = 1.0
                    cuts.append(("clique", row, 1.0))
                    if cut_details is not None:
                        cut_details.append({
                            "type"      : "clique",
                            "clique"    : list(clique),
                            "slot_name" : inst.slot_names[k],
                            "lhs_val"   : float(lhs_val),
                            "violation" : float(lhs_val - 1.0),
                        })
        return cuts

    # --- override para incluir clique na contagem do log ---
    def _print_cuts(self, cut_details: list, node, round_num: int):
        pad = "    " * node.depth
        n_gomory = sum(1 for d in cut_details if d.get("type") == "gomory")
        n_cover  = sum(1 for d in cut_details if d.get("type") == "cover")
        n_clique = sum(1 for d in cut_details if d.get("type") == "clique")
        n_total  = n_gomory + n_cover + n_clique
        var_names= self.model.var_names
        n        = len(self.model.c)
        p        = pad + "    "
        print(f"\n{p}[B&C] No {node.id} - rodada {round_num}: {n_total} corte(s)  "
              f"[{n_gomory} Gomory, {n_cover} Cover, {n_clique} Clique]")
        for d in cut_details:
            t = d.get("type")
            if t == "gomory_tableau":
                self._print_gomory_tableau(d, pad=pad)
            elif t == "gomory":
                self._print_gomory_cut(d, var_names, n, pad=pad)
            elif t == "cover":
                self._print_cover_cut(d, var_names, n, pad=pad)
            elif t == "clique":
                disc_names = self.inst.disc_names
                names = " + ".join(disc_names[i] for i in d["clique"])
                print(f"{p}>> Clique [{names}] no horario {d['slot_name']}  "
                      f"(soma={d['lhs_val']:.4f})  violacao={d['violation']:.4f}")
                print()


# ======================================================================
# Exibicao e verificacao da solucao
# ======================================================================

def print_schedule(
    inst: TimetablingInstance,
    x: np.ndarray,
    obj: float,
    label: str = "",
) -> None:
    nd, nr, nt = inst.nd, inst.nr, inst.nt
    print(f"\n{'=' * 62}")
    if label:
        print(f"  {label}")
    print(f"  Satisfacao total: {obj:.4f}")
    print()
    print(f"  {'Disciplina':<12} {'Sala':<8} {'Horario':<8}  Satisf.")
    print(f"  {'-' * 44}")
    for i in range(nd):
        for j in range(nr):
            for k in range(nt):
                if x[inst.idx(i, j, k)] > 0.5:
                    s = inst.sat[i][j][k] or 0.0
                    print(f"  {inst.disc_names[i]:<12} {inst.room_names[j]:<8} "
                          f"{inst.slot_names[k]:<8}  {s:.4f}")
    print("=" * 62)


def verify_solution(inst: TimetablingInstance, x: np.ndarray) -> list[str]:
    """Verifica restricoes; retorna lista de violacoes (vazia = OK)."""
    nd, nr, nt = inst.nd, inst.nr, inst.nt
    errors: list[str] = []

    # (1) alocacao unica
    for i in range(nd):
        assigned = sum(round(x[inst.idx(i, j, k)]) for j in range(nr) for k in range(nt))
        if assigned != 1:
            errors.append(f"{inst.disc_names[i]}: {assigned} alocacoes (esperado 1)")

    # (2) conflito de sala
    for j in range(nr):
        for k in range(nt):
            count = sum(round(x[inst.idx(i, j, k)]) for i in range(nd))
            if count > 1:
                errors.append(f"Sala {inst.room_names[j]} / {inst.slot_names[k]}: {count} disciplinas")

    # (5) precedencia
    for (i_from, i_to) in inst.precedence:
        t_from = sum(
            (k + 1) * round(x[inst.idx(i_from, j, k)])
            for j in range(nr) for k in range(nt)
        )
        t_to = sum(
            (k + 1) * round(x[inst.idx(i_to, j, k)])
            for j in range(nr) for k in range(nt)
        )
        if t_from >= t_to:
            errors.append(
                f"Precedencia: {inst.disc_names[i_from]}(T{t_from}) "
                f">= {inst.disc_names[i_to]}(T{t_to})"
            )

    # (6) anti-conflito
    for (i1, i2) in inst.anticonflict:
        for k in range(nt):
            a1 = sum(round(x[inst.idx(i1, j, k)]) for j in range(nr))
            a2 = sum(round(x[inst.idx(i2, j, k)]) for j in range(nr))
            if a1 + a2 > 1:
                errors.append(
                    f"Anti-conflito {inst.disc_names[i1]}-{inst.disc_names[i2]} "
                    f"em {inst.slot_names[k]}"
                )

    return errors


# ======================================================================
# OR-Tools CP-SAT
# ======================================================================

def solve_ortools(inst: TimetablingInstance) -> tuple[float | None, np.ndarray | None]:
    if not HAS_ORTOOLS:
        return None, None

    from ortools.sat.python import cp_model
    nd, nr, nt = inst.nd, inst.nr, inst.nt

    model = cp_model.CpModel()

    # variaveis booleanas (apenas para alocacoes viaveis)
    xv: dict[tuple[int,int,int], object] = {}
    for i in range(nd):
        for j in range(nr):
            for k in range(nt):
                if inst.sat[i][j][k] is not None:
                    xv[i, j, k] = model.NewBoolVar(f"x_{i}_{j}_{k}")

    # (1) alocacao unica
    for i in range(nd):
        model.AddExactlyOne(
            xv[i, j, k] for j in range(nr) for k in range(nt) if (i, j, k) in xv
        )

    # (2) conflito de sala
    for j in range(nr):
        for k in range(nt):
            model.AddAtMostOne(
                xv[i, j, k] for i in range(nd) if (i, j, k) in xv
            )

    # (5) precedencia via variaveis auxiliares de horario
    for (i_from, i_to) in inst.precedence:
        t_from = model.NewIntVar(1, nt, f"t_from_{i_from}")
        t_to   = model.NewIntVar(1, nt, f"t_to_{i_to}")
        for j in range(nr):
            for k in range(nt):
                if (i_from, j, k) in xv:
                    model.Add(t_from == k + 1).OnlyEnforceIf(xv[i_from, j, k])
                if (i_to, j, k) in xv:
                    model.Add(t_to == k + 1).OnlyEnforceIf(xv[i_to, j, k])
        model.Add(t_from < t_to)

    # (6) anti-conflito
    for (i1, i2) in inst.anticonflict:
        for k in range(nt):
            pair = (
                [xv[i1, j, k] for j in range(nr) if (i1, j, k) in xv] +
                [xv[i2, j, k] for j in range(nr) if (i2, j, k) in xv]
            )
            if pair:
                model.AddAtMostOne(pair)

    # objetivo (escalonado para inteiro)
    SCALE = 10000
    obj_terms = []
    for (i, j, k), var in xv.items():
        s = inst.sat[i][j][k]
        if s is not None:
            obj_terms.append(int(round(s * SCALE)) * var)
    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        obj = solver.ObjectiveValue() / SCALE
        x_out = np.zeros(inst.n_vars())
        for (i, j, k), var in xv.items():
            x_out[inst.idx(i, j, k)] = float(solver.Value(var))
        return obj, x_out

    return None, None


# ======================================================================
# Main
# ======================================================================

def run_instance(
    label: str,
    inst: TimetablingInstance,
    verbose_bc: bool = True,
    time_limit: float | None = None,
):
    print("\n" + "=" * 62)
    print(f"  {label}")
    print("=" * 62)

    # --- dados da instancia ---
    print(f"\n  Disciplinas ({inst.nd}):")
    for i in range(inst.nd):
        feasible_rooms = [
            inst.room_names[j] for j in range(inst.nr)
            if any(inst.sat[i][j][k] is not None for k in range(inst.nt))
        ]
        print(f"    {inst.disc_names[i]:<8} matriculas={inst.enrollments[i]:>3}  "
              f"salas: {', '.join(feasible_rooms)}")

    print(f"\n  Salas ({inst.nr}):")
    for j in range(inst.nr):
        print(f"    {inst.room_names[j]}: capacidade {inst.capacities[j]}")

    print(f"\n  Horarios ({inst.nt}): {', '.join(inst.slot_names)}")

    for (a, b) in inst.precedence:
        print(f"\n  Precedencia : {inst.disc_names[a]} deve preceder {inst.disc_names[b]}")
    for (a, b) in inst.anticonflict:
        print(f"  Anti-conflito: {inst.disc_names[a]} nao divide horario com {inst.disc_names[b]}")

    # --- grafo de conflito e cliques ---
    adj    = build_conflict_graph(inst)
    cliques= find_maximal_cliques(adj)
    print(f"\n  Grafo de conflito (arestas):")
    has_edge = False
    for i in range(inst.nd):
        for ip in sorted(adj[i]):
            if ip > i:
                print(f"    {inst.disc_names[i]} -- {inst.disc_names[ip]}")
                has_edge = True
    if not has_edge:
        print("    (sem arestas)")
    print(f"\n  Cliques maximais ({len(cliques)}):")
    for c in cliques:
        print(f"    {{ {', '.join(inst.disc_names[i] for i in c)} }}")

    # --- modelo MIP ---
    model = build_model(inst)
    n_ub  = len(model.b_ub) if model.b_ub is not None else 0
    n_eq  = len(model.b_eq) if model.b_eq is not None else 0
    print(f"\n  Modelo MIP: {model.n_vars} variaveis | "
          f"{n_ub} restricoes <= | {n_eq} restricoes =")

    # --- B&C ---
    print(f"\n{'=' * 62}")
    print("  BRANCH-AND-CUT (depth-first, clique + Gomory)")
    print("=" * 62)

    bc = TimetablingBranchAndCut(
        inst, model,
        strategy       = "dfs",
        branching      = "most_infeasible",
        cut_types      = ["clique", "gomory"],
        max_cut_rounds = 3,
        time_limit     = time_limit,
        verbose        = verbose_bc,
    )
    bc_result = bc.solve()

    bc_obj  = None
    bc_x    = None
    if bc_result.status == "optimal":
        bc_obj = bc_result.obj_value
        bc_x   = bc_result.x
        print_schedule(inst, bc_x, bc_obj, "B&C | Solucao otima")
        errs = verify_solution(inst, bc_x)
        if errs:
            print("  VIOLACOES DETECTADAS:")
            for e in errs:
                print(f"    {e}")
        else:
            print("  Todas as restricoes satisfeitas.")
        print(f"  Nos explorados : {bc_result.nodes_explored}")
        print(f"  Tempo          : {bc_result.elapsed:.2f}s")
        n_cuts = len(bc._cut_log)
        print(f"  Cortes gerados : {n_cuts}")
        if n_cuts == 0 and bc_result.nodes_explored == 1:
            print("  (LP raiz ja era inteira: formulacao com = e <= resulta em matriz")
            print("   TU para esses dados, sem necessidade de ramificacao ou cortes)")
        types_used: dict[str, int] = {}
        for entry in bc._cut_log:
            t = entry["type"]
            types_used[t] = types_used.get(t, 0) + 1
        for t, cnt in sorted(types_used.items()):
            print(f"    {t}: {cnt}")
    elif bc_result.status == "timeout":
        print(f"  B&C: TIMEOUT apos {bc_result.nodes_explored} nos")
        if bc_result.obj_value is not None:
            bc_obj = bc_result.obj_value
            bc_x   = bc_result.x
            print(f"  Melhor incumbente: {bc_obj:.4f}")
    else:
        print(f"  B&C: {bc_result.status}")

    # --- OR-Tools ---
    print(f"\n{'=' * 62}")
    print("  OR-TOOLS CP-SAT")
    print("=" * 62)

    ort_obj = None
    ort_x   = None
    if HAS_ORTOOLS:
        ort_obj, ort_x = solve_ortools(inst)
        if ort_x is not None:
            print_schedule(inst, ort_x, ort_obj, "CP-SAT | Solucao")
            errs = verify_solution(inst, ort_x)
            if not errs:
                print("  Todas as restricoes satisfeitas.")
            else:
                for e in errs:
                    print(f"  VIOLACAO: {e}")
        else:
            print("  CP-SAT nao encontrou solucao viavel.")
    else:
        print("\n  OR-Tools nao instalado.")
        print("  Para instalar: pip install ortools")

    # --- comparacao ---
    print(f"\n{'=' * 62}")
    print("  COMPARACAO")
    print("=" * 62)
    if bc_obj is not None:
        print(f"  B&C      obj={bc_obj:.4f}  ({bc_result.nodes_explored} nos)")
    else:
        print("  B&C      infeasivel ou timeout")
    if ort_obj is not None:
        print(f"  CP-SAT   obj={ort_obj:.4f}")
        if bc_obj is not None:
            gap = abs(bc_obj - ort_obj)
            print(f"  Gap      {gap:.6f}  ({'IGUAL' if gap < 1e-4 else 'DIFERENTE'})")
    else:
        print("  CP-SAT   nao disponivel")
    print()


if __name__ == "__main__":
    run_instance(
        "INSTANCIA PEQUENA (4 disc, 3 salas, 4 horarios)",
        make_small_instance(),
        verbose_bc=True,
        time_limit=None,
    )
    run_instance(
        "INSTANCIA GRANDE (8 disc, 5 salas, 6 horarios)",
        make_large_instance(),
        verbose_bc=False,
        time_limit=60.0,   # limite de 60s; CP-SAT resolve exatamente
    )
