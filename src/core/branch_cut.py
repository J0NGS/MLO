"""Branch-and-Cut: extends BranchAndBound with automatic cut generation."""
from __future__ import annotations
import math
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
        #   dfs        ; pilha (lifo);
        #   bfs        ; fila (fifo);
        #   best_first ; heap de prioridade; sempre processa o nó com melhor bound lp; mais eficiente na prática
        #              ; no best_first: pra min, prioridade = menor obj do pai; pra max, prioridade = maior obj do pai
        #              ; isso garante que os nós mais promissores sejam explorados antes;
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

        for round_num in range(1, self.max_cut_rounds + 1):
            n_before = len(node.cut_lhs)
            cut_details = [] if self.verbose else None
            new_cuts = self._generate_cuts(current_lp.x, node, current_lp, cut_details=cut_details)
            if not new_cuts:
                # ponto de decisão; nenhum corte útil encontrado; sai do loop e vai ramificar
                break

            for cut_type, lhs, rhs in new_cuts:
                # acumula os cortes no nó; cada filho vai herdar esses cortes ao nascer
                node.cut_lhs.append(lhs)
                node.cut_rhs.append(rhs)
                total_cuts += 1
                self._cut_log.append({"node": node.id, "type": cut_type, "rhs": float(rhs)})

            if self.verbose and cut_details:
                self._print_cuts(cut_details, node, round_num)

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

    def _generate_cuts(self, x: np.ndarray, node: Node, lp_result=None, cut_details: list | None = None) -> list[tuple[str, np.ndarray, float]]:
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
                                                 slack=slack, node_bounds=node.bounds,
                                                 cut_details=cut_details)]

        if "cover" in self.cut_types:
            # cover; detecta subconjuntos de variáveis binárias que somam mais que a capacidade; específico pra knapsack
            new_cover = list(cover_cuts(x, self.model))
            cuts += [("cover", lhs, rhs) for lhs, rhs in new_cover]
            if cut_details is not None:
                var_names = self.model.var_names
                n = len(x)
                for lhs, rhs in new_cover:
                    cut_details.append({
                        "type": "cover",
                        "cut_lhs": lhs.copy(),
                        "cut_rhs": float(rhs),
                        "violation": float(lhs @ x - rhs),
                    })

        return cuts

    # ------------------------------------------------------------------
    # Logging de cortes (verbose)
    # ------------------------------------------------------------------

    def _print_cuts(self, cut_details: list, node: Node, round_num: int):
        pad = "    " * node.depth
        n_gomory = sum(1 for d in cut_details if d.get("type") == "gomory")
        n_cover  = sum(1 for d in cut_details if d.get("type") == "cover")
        n_total  = n_gomory + n_cover
        var_names = self.model.var_names
        n = len(self.model.c)
        print(f"\n{pad}    [B&C] No {node.id} - rodada {round_num}: {n_total} corte(s)  "
              f"[{n_gomory} Gomory, {n_cover} Cover]")
        for d in cut_details:
            t = d.get("type")
            if t == "gomory_tableau":
                self._print_gomory_tableau(d, pad=pad)
            elif t == "gomory":
                self._print_gomory_cut(d, var_names, n, pad=pad)
            elif t == "cover":
                self._print_cover_cut(d, var_names, n, pad=pad)

    def _print_gomory_tableau(self, d: dict, pad: str = ""):
        var_names = d["var_names"]
        n         = d["n_orig"]
        m         = d["m_slack"]
        tableau   = d["full_tableau"]
        basic_idx = d["basic_idx"]
        x_B       = d["x_B"]
        frac_set  = {k for k, _ in d["frac_rows"]}
        p         = pad + "    "

        def var_label(idx: int) -> str:
            if idx < n:
                return var_names[idx] if idx < len(var_names) else f"x{idx}"
            return f"s{idx - n + 1}"

        col_labels = [var_names[j] if j < len(var_names) else f"x{j}" for j in range(n)]
        col_labels += [f"s{j + 1}" for j in range(m)]
        col_w  = max(8, max((len(l) for l in col_labels), default=4) + 2)
        base_w = max(8, max((len(var_label(bi)) + 2 for bi in basic_idx), default=6))

        print(f"{p}Tableau simplex (solucao fracionaria):")
        header = f"{p}{'Base':<{base_w}} {'Valor':>10} {'Fracao':>8}  |"
        for l in col_labels:
            header += f"  {l:>{col_w}}"
        print(header)
        sep = f"{p}{'-' * base_w} {'-' * 10} {'-' * 8}  +"
        for _ in col_labels:
            sep += "-" * (col_w + 2)
        print(sep)

        for k, bi in enumerate(basic_idx):
            marker = "*" if k in frac_set else " "
            label  = var_label(bi) + marker
            val    = x_B[k]
            f_val  = val - math.floor(val)
            f_str  = f"{f_val:.6f}" if k in frac_set else "    -   "
            row = f"{p}{label:<{base_w}} {val:>10.6f} {f_str:>8}  |"
            for j in range(n + m):
                row += f"  {tableau[k, j]:>{col_w}.4f}"
            print(row)

        if frac_set:
            print(f"{p}(* variavel inteira fracionaria => gera corte de Gomory)")
        print()

    def _print_gomory_cut(self, d: dict, var_names: list, n: int, pad: str = ""):
        var_name  = d["var_name"]
        xB_k      = d["xB_k"]
        f_k       = d["f_k"]
        cut_lhs   = d["cut_lhs"]
        cut_rhs   = d["cut_rhs"]
        violation = d["violation"]
        p         = pad + "    "

        nonzero = [(j, cut_lhs[j]) for j in range(n) if abs(cut_lhs[j]) > 1e-8]
        terms = "  ".join(
            f"({c:+.4f})*{var_names[j] if j < len(var_names) else f'x{j}'}"
            for j, c in nonzero
        ) if nonzero else "0"

        print(f"{p}>> Gomory  ({var_name} = {xB_k:.6f},  f = {f_k:.6f}):")
        print(f"{p}   {terms}  <=  {cut_rhs:.6f}")
        print(f"{p}   violacao: {violation:.6f}")
        print()

    def _print_cover_cut(self, d: dict, var_names: list, n: int, pad: str = ""):
        cut_lhs   = d["cut_lhs"]
        cut_rhs   = d["cut_rhs"]
        violation = d["violation"]
        p         = pad + "    "
        cover_vars = [
            var_names[j] if j < len(var_names) else f"x{j}"
            for j in range(n) if cut_lhs[j] > 0.5
        ]
        print(f"{p}>> Cover: {' + '.join(cover_vars)}  <=  {cut_rhs:.0f}")
        print(f"{p}   violacao: {violation:.6f}")
        print()
