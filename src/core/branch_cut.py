"""Branch-and-Cut: extends BranchAndBound with automatic cut generation."""
from __future__ import annotations
import numpy as np

from .branch_bound import BranchAndBound
from .node import Node
from .relaxation import LPResult
from .cuts import gomory_cuts, cover_cuts


# herda tudo do b&b; a única diferença real está no _process_node
class BranchAndCut(BranchAndBound):
    """
    B&C engine. After solving a node LP, attempts to add cuts before branching.

    cut_types: list of "gomory" | "cover"
    max_cut_rounds: how many separation rounds per node
    """

    def __init__(self, *args, cut_types: list[str] | None = None, max_cut_rounds: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        # estratégias de exploração herdadas do b&b (parâmetro strategy):
        #   dfs        ; pilha (lifo); vai fundo antes de voltar; usa menos memória; bom pra achar incumbente rápido
        #   bfs        ; fila (fifo); explora nível por nível; guarda muitos nós em memória
        #   best_first ; heap de prioridade; sempre processa o nó com melhor bound lp; mais eficiente na prática
        #              ; no best_first: pra min, prioridade = menor obj do pai; pra max, prioridade = maior obj do pai
        #              ; isso garante que os nós mais promissores sejam explorados antes; menos podas desperdiçadas
        self.cut_types = cut_types or ["gomory"]  # quais geradores de corte usar
        self.max_cut_rounds = max_cut_rounds       # quantas rodadas de corte por nó no máximo
        self._cut_log: list[dict] = []

    def _process_node(self, node: Node, lp: LPResult) -> tuple[str, int]:
        """Override: try cuts before declaring 'branched'."""
        # mesmas três podas do b&b puro; se cair em qualquer uma delas, nem chega no loop de cortes
        if lp.status == "infeasible":
            return "pruned_infeasible", 0

        if self._incumbent is not None:
            if self.model.sense == "min" and lp.obj_value >= self._incumbent - 1e-8:
                return "pruned_bound", 0
            if self.model.sense == "max" and lp.obj_value <= self._incumbent + 1e-8:
                return "pruned_bound", 0

        if self._is_integer(lp.x):
            self._update_incumbent(lp.obj_value, lp.x)
            return "pruned_integer", 0

        # [diferença do b&b] aqui começa o loop de cortes; no b&b puro iria direto pro branching
        total_cuts = 0
        current_lp = lp

        for _ in range(self.max_cut_rounds):
            n_before = len(node.cut_lhs)
            new_cuts = self._generate_cuts(current_lp.x, node, current_lp)
            if not new_cuts:
                # ponto de decisão; nenhum corte útil encontrado; sai do loop e vai ramificar
                break

            for cut_type, lhs, rhs in new_cuts:
                # acumula os cortes no nó; cada filho vai herdar esses cortes ao nascer
                node.cut_lhs.append(lhs)
                node.cut_rhs.append(rhs)
                total_cuts += 1
                self._cut_log.append({"node": node.id, "type": cut_type, "rhs": float(rhs)})

            # re-resolve o lp com os novos cortes; tenta apertar o bound antes de ramificar
            test_lp = self._solve_node(node)

            if test_lp.status == "infeasible":
                # ponto de decisão; corte gerou inviabilidade; descarta essa rodada e ramifica no lp anterior
                del node.cut_lhs[n_before:]
                del node.cut_rhs[n_before:]
                total_cuts -= len(new_cuts)
                break

            current_lp = test_lp

            if self._incumbent is not None:
                # ponto de decisão; corte pode ter fechado a janela de bound; poda sem ramificar
                if self.model.sense == "min" and current_lp.obj_value >= self._incumbent - 1e-8:
                    return "pruned_bound", total_cuts
                if self.model.sense == "max" and current_lp.obj_value <= self._incumbent + 1e-8:
                    return "pruned_bound", total_cuts

            if self._is_integer(current_lp.x):
                # ponto de decisão; corte resolveu a fracionariedade; incumbente atualizado sem ramificar
                self._update_incumbent(current_lp.obj_value, current_lp.x)
                node.lp_obj = current_lp.obj_value
                node.lp_x = current_lp.x
                return "pruned_integer", total_cuts

        # lp do nó atualizado com todos os cortes aplicados; agora sim vai ramificar
        node.lp_obj = current_lp.obj_value
        node.lp_x = current_lp.x
        return "branched", total_cuts

    def _generate_cuts(self, x: np.ndarray, node: Node, lp_result=None) -> list[tuple[str, np.ndarray, float]]:
        """Run all enabled cut generators; return (cut_type, lhs, rhs) triples."""
        cuts: list[tuple[str, np.ndarray, float]] = []

        # monta A_ub incluindo cortes que já foram adicionados a esse nó em rodadas anteriores
        A_ub_eff = self.model.A_ub
        b_ub_eff = self.model.b_ub
        if node.cut_lhs:
            extra = np.array(node.cut_lhs)
            extra_rhs = np.array(node.cut_rhs)
            if A_ub_eff is not None:
                A_ub_eff = np.vstack([A_ub_eff, extra])
                b_ub_eff = np.concatenate([b_ub_eff, extra_rhs])
            else:
                A_ub_eff = extra
                b_ub_eff = extra_rhs

        slack = lp_result.slack if lp_result is not None else None

        if "gomory" in self.cut_types:
            # gomory; derivado da tabela do simplex; corta a solução fracionária sem remover inteiros válidos
            cuts += [("gomory", lhs, rhs)
                     for lhs, rhs in gomory_cuts(x, self.model, A_ub_eff, b_ub_eff,
                                                 slack=slack, node_bounds=node.bounds)]

        if "cover" in self.cut_types:
            # cover; detecta subconjuntos de variáveis binárias que somam mais que a capacidade; específico pra knapsack
            cuts += [("cover", lhs, rhs) for lhs, rhs in cover_cuts(x, self.model)]

        return cuts
