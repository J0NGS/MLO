"""Branch-and-Cut: extends BranchAndBound with automatic cut generation."""
from __future__ import annotations
import numpy as np

from .branch_bound import BranchAndBound
from .node import Node
from .relaxation import LPResult
from .cuts import gomory_cuts, cover_cuts


class BranchAndCut(BranchAndBound):
    """
    B&C engine. After solving a node LP, attempts to add cuts before branching.

    cut_types: list of "gomory" | "cover"
    max_cut_rounds: how many separation rounds per node
    """

    def __init__(self, *args, cut_types: list[str] | None = None, max_cut_rounds: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.cut_types = cut_types or ["gomory"]
        self.max_cut_rounds = max_cut_rounds

    def _process_node(self, node: Node, lp: LPResult) -> tuple[str, int]:
        """Override: try cuts before declaring 'branched'."""
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

        # --- Cut generation loop ---
        total_cuts = 0
        current_lp = lp

        for _ in range(self.max_cut_rounds):
            n_before = len(node.cut_lhs)
            new_cuts = self._generate_cuts(current_lp.x, node, current_lp)
            if not new_cuts:
                break

            for lhs, rhs in new_cuts:
                node.cut_lhs.append(lhs)
                node.cut_rhs.append(rhs)
                total_cuts += 1

            # Re-solve with new cuts
            test_lp = self._solve_node(node)

            if test_lp.status == "infeasible":
                # Roll back: cuts may be invalid (e.g. pure-IP Gomory applied to PLIM).
                # Discard the offending round and branch on the last valid LP.
                del node.cut_lhs[n_before:]
                del node.cut_rhs[n_before:]
                total_cuts -= len(new_cuts)
                break

            current_lp = test_lp

            if self._incumbent is not None:
                if self.model.sense == "min" and current_lp.obj_value >= self._incumbent - 1e-8:
                    return "pruned_bound", total_cuts
                if self.model.sense == "max" and current_lp.obj_value <= self._incumbent + 1e-8:
                    return "pruned_bound", total_cuts

            if self._is_integer(current_lp.x):
                self._update_incumbent(current_lp.obj_value, current_lp.x)
                node.lp_obj = current_lp.obj_value
                node.lp_x = current_lp.x
                return "pruned_integer", total_cuts

        # Update node LP info after cuts
        node.lp_obj = current_lp.obj_value
        node.lp_x = current_lp.x
        return "branched", total_cuts

    def _generate_cuts(self, x: np.ndarray, node: Node, lp_result=None) -> list[tuple[np.ndarray, float]]:
        """Run all enabled cut generators and return new violated cuts."""
        cuts = []

        # Build effective A_ub including cuts already at this node
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
            cuts += gomory_cuts(x, self.model, A_ub_eff, b_ub_eff,
                                slack=slack, node_bounds=node.bounds)

        if "cover" in self.cut_types:
            cuts += cover_cuts(x, self.model)

        return cuts
