"""Generic Branch-and-Bound solver."""
from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from .model import MIPModel
from .node import Node, NodeLog
from .relaxation import solve_relaxation, LPResult


# resultado final do solver; o que o b&b devolve pra quem chamou
@dataclass
class BBResult:
    status: str                      # "optimal", "infeasible", "timeout"
    obj_value: Optional[float]
    x: Optional[np.ndarray]
    nodes_explored: int
    elapsed: float
    log: list[NodeLog] = field(default_factory=list)


# motor de busca; não sabe nada sobre projetos, só lida com o modelo
class BranchAndBound:
    """
    Generic B&B engine.

    Parameters
    ----------
    model              : MIPModel
    strategy           : "dfs" | "bfs" | "best_first"
    branching          : "first_fractional" | "most_infeasible"
    time_limit         : seconds (None = no limit)
    initial_incumbent  : known feasible objective value (warm start)
    initial_x          : solution vector corresponding to initial_incumbent
    """

    def __init__(
        self,
        model: MIPModel,
        strategy: str = "best_first",
        branching: str = "most_infeasible",
        time_limit: Optional[float] = None,
        initial_incumbent: Optional[float] = None,
        initial_x: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.strategy = strategy
        self.branching = branching
        self.time_limit = time_limit
        self.verbose = verbose

        self._node_counter = 0
        self._incumbent: Optional[float] = initial_incumbent
        self._best_x: Optional[np.ndarray] = initial_x
        self._log: list[NodeLog] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> BBResult:
        # loop principal: cria a raiz, coloca na fila e processa nó a nó
        t0 = time.perf_counter()

        root = self._make_root()
        queue = self._make_queue()
        self._enqueue(queue, root, None)

        nodes_explored = 0

        while queue:
            if self.time_limit and (time.perf_counter() - t0) > self.time_limit:
                return BBResult(
                    "timeout", self._incumbent, self._best_x,
                    nodes_explored, time.perf_counter() - t0, self._log,
                )

            node = self._dequeue(queue)
            nodes_explored += 1

            lp = self._solve_node(node)
            node.lp_status = lp.status
            node.lp_obj = lp.obj_value
            node.lp_x = lp.x

            decision, cuts_added = self._process_node(node, lp)
            node.decision = decision

            self._log.append(NodeLog(
                node_id=node.id,
                parent_id=node.parent_id,
                depth=node.depth,
                branch_var_name=self.model.var_names[node.branch_var] if node.branch_var is not None else None,
                branch_dir=node.branch_dir,
                branch_val=node.branch_val,
                lp_status=lp.status,
                lp_obj=lp.obj_value,
                best_incumbent=self._incumbent,
                decision=decision,
                cuts_added=cuts_added,
            ))

            self._print_node(node, cuts_added)

            if decision == "branched":
                # Use post-cut LP solution when available (B&C updates node.lp_x after cuts)
                branch_x = node.lp_x if node.lp_x is not None else lp.x
                branch_obj = node.lp_obj if node.lp_obj is not None else lp.obj_value
                children = self._branch(node, branch_x)
                for child in children:
                    self._enqueue(queue, child, branch_obj)

        elapsed = time.perf_counter() - t0
        if self.verbose:
            self._print_tree()
        if self._incumbent is not None:
            return BBResult("optimal", self._incumbent, self._best_x, nodes_explored, elapsed, self._log)
        return BBResult("infeasible", None, None, nodes_explored, elapsed, self._log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_root(self) -> Node:
        # raiz usa os bounds originais; nenhum branching foi feito ainda
        self._node_counter = 0
        node = Node(
            id=0,
            parent_id=None,
            depth=0,
            bounds=list(self.model.bounds),
        )
        return node

    def _make_queue(self):
        # tipo da estrutura define a ordem de exploração da árvore
        if self.strategy == "dfs":
            return []           # use as stack (append/pop)
        elif self.strategy == "bfs":
            return deque()
        else:                   # best_first: list of (priority, node)
            return []

    def _enqueue(self, queue, node: Node, parent_lb: Optional[float]):
        if self.strategy == "dfs":
            queue.append(node)
        elif self.strategy == "bfs":
            queue.append(node)
        else:
            import heapq
            # best-first: menor bound pra min, maior pra max
            priority = -(parent_lb or 0) if self.model.sense == "max" else (parent_lb or 0)
            heapq.heappush(queue, (priority, node.id, node))

    def _dequeue(self, queue):
        if self.strategy == "dfs":
            return queue.pop()
        elif self.strategy == "bfs":
            return queue.popleft()
        else:
            import heapq
            _, _, node = heapq.heappop(queue)
            return node

    def _solve_node(self, node: Node) -> LPResult:
        # relaxação do nó = modelo original com os bounds ajustados desse nó
        extra_lhs = np.array(node.cut_lhs) if node.cut_lhs else None
        extra_rhs = np.array(node.cut_rhs) if node.cut_rhs else None
        return solve_relaxation(self.model, extra_lhs, extra_rhs, node.bounds)

    def _process_node(self, node: Node, lp: LPResult) -> tuple[str, int]:
        """Returns (decision, cuts_added). Subclasses may override for B&C."""
        if lp.status == "infeasible":
            # relaxação inviável = nenhuma solução inteira pode existir aqui
            return "pruned_infeasible", 0

        # poda por bound; lp já é pior que o melhor inteiro conhecido
        if self._incumbent is not None:
            if self.model.sense == "min" and lp.obj_value >= self._incumbent - 1e-8:
                return "pruned_bound", 0
            if self.model.sense == "max" and lp.obj_value <= self._incumbent + 1e-8:
                return "pruned_bound", 0

        # solução do lp já é inteira — atualiza incumbente e não precisa ramificar
        if self._is_integer(lp.x):
            self._update_incumbent(lp.obj_value, lp.x)
            return "pruned_integer", 0

        return "branched", 0

    def _is_integer(self, x: np.ndarray, tol: float = 1e-5) -> bool:
        # tolerância pra erro numérico — parte fracionária fora do intervalo (tol, 1-tol) é fracionária
        for i in self.model.integer_indices:
            frac = x[i] - math.floor(x[i])
            if tol < frac < 1 - tol:
                return False
        return True

    def _update_incumbent(self, obj: float, x: np.ndarray):
        if self._incumbent is None:
            self._incumbent = obj
            self._best_x = x.copy()
        elif self.model.sense == "min" and obj < self._incumbent - 1e-8:
            self._incumbent = obj
            self._best_x = x.copy()
        elif self.model.sense == "max" and obj > self._incumbent + 1e-8:
            self._incumbent = obj
            self._best_x = x.copy()

    def _pick_branch_var(self, x: np.ndarray) -> Optional[int]:
        # escolhe qual variável fracionária vai ser dividida em dois subproblemas
        int_idx = self.model.integer_indices
        fracs = [(i, x[i] - math.floor(x[i])) for i in int_idx]
        fracs = [(i, f) for i, f in fracs if 1e-5 < f < 1 - 1e-5]
        if not fracs:
            return None

        if self.branching == "first_fractional":
            return fracs[0][0]
        else:  # most_infeasible — fração mais próxima de 0.5 gera dois filhos mais equilibrados
            return max(fracs, key=lambda t: min(t[1], 1 - t[1]))[0]

    def _branch(self, node: Node, x: np.ndarray) -> list[Node]:
        # cria dois filhos: down → xj <= floor(v), up → xj >= ceil(v)
        var = self._pick_branch_var(x)
        if var is None:
            return []

        val = x[var]
        floor_val = math.floor(val)
        ceil_val = math.ceil(val)

        children = []
        for direction, new_bound_val in [("down", floor_val), ("up", ceil_val)]:
            self._node_counter += 1
            child_bounds = list(node.bounds)
            lb, ub = child_bounds[var]

            if direction == "down":
                child_bounds[var] = (lb, new_bound_val)
            else:
                child_bounds[var] = (new_bound_val, ub)

            child = Node(
                id=self._node_counter,
                parent_id=node.id,
                depth=node.depth + 1,
                branch_var=var,
                branch_dir=direction,
                branch_val=new_bound_val,
                bounds=child_bounds,
                cut_lhs=list(node.cut_lhs),
                cut_rhs=list(node.cut_rhs),
            )
            children.append(child)

        return children

    def _print_node(self, node: Node, cuts_added: int = 0):
        if not self.verbose:
            return
        indent    = "    " * node.depth
        inc       = f"{self._incumbent:.4f}" if self._incumbent is not None else "---"
        lb        = f"{node.lp_obj:.4f}" if node.lp_obj is not None else "infeas"
        if node.branch_var is not None:
            branch = f"{self.model.var_names[node.branch_var]}({node.branch_dir[:2]})"
        else:
            branch = "raiz"
        cuts_info = f" +{cuts_added}c" if cuts_added > 0 else ""
        dec_map = {
            "pruned_integer"    : "INT",
            "pruned_bound"      : "BND",
            "pruned_infeasible" : "INF",
            "branched"          : "RAM",
        }
        dec = dec_map.get(node.decision, node.decision or "?")
        print(f"{indent}[No {node.id:3d}] LP={lb:>8}  inc={inc}  "
              f"{branch}{cuts_info}  =>  {dec}")

    def _print_tree(self) -> None:
        """Reconstroi e imprime a arvore B&B completa apos o solve."""
        if not self._log:
            return

        by_id: dict[int, NodeLog] = {e.node_id: e for e in self._log}
        kids : dict[int, list[int]] = {e.node_id: [] for e in self._log}
        for e in self._log:
            if e.parent_id is not None and e.parent_id in kids:
                kids[e.parent_id].append(e.node_id)

        # exibe filhos em ordem logica: down primeiro, up depois
        for nid in kids:
            kids[nid].sort(key=lambda cid: (by_id[cid].branch_dir or "", cid))

        n    = len(self._log)
        opt  = f"  otimo={self._incumbent:.4f}" if self._incumbent is not None else ""
        print(f"\n{'=' * 60}")
        print(f"  ARVORE B&B  ({n} nos{opt})")
        print("=" * 60)

        dec_map = {
            "pruned_integer"    : "INT",
            "pruned_bound"      : "BND",
            "pruned_infeasible" : "INF",
            "branched"          : "RAM",
        }

        def _fmt(e: NodeLog) -> str:
            lb    = f"{e.lp_obj:.4f}" if e.lp_obj is not None else "infeas"
            inc   = f"{e.best_incumbent:.4f}" if e.best_incumbent is not None else "---"
            bname = e.branch_var_name or "raiz"
            bdir  = f"({e.branch_dir[:2]})" if e.branch_dir else ""
            cuts  = f" +{e.cuts_added}c" if e.cuts_added else ""
            dec   = dec_map.get(e.decision, e.decision or "?")
            return f"[No {e.node_id:3d}] LP={lb:>8}  inc={inc}  {bname}{bdir}{cuts}  {dec}"

        def _sub(nid: int, prefix: str = "", is_last: bool = True) -> None:
            e = by_id[nid]
            if prefix == "":
                print(_fmt(e))
            else:
                conn = "└──" if is_last else "├──"
                print(f"{prefix}{conn} {_fmt(e)}")
            children = kids.get(nid, [])
            for i, kid in enumerate(children):
                ext = "    " if is_last else "│   "
                _sub(kid, prefix + ext, i == len(children) - 1)

        _sub(0)
        print()
