from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    depth: int

    # Branching info (what constraint was added to reach this node)
    branch_var: Optional[int] = None
    branch_dir: Optional[str] = None   # "down" (<=floor) or "up" (>=ceil)
    branch_val: Optional[float] = None

    # Per-node variable bounds (copy of parent bounds, then tightened)
    bounds: list[tuple[float | None, float | None]] = field(default_factory=list)

    # Extra cuts accumulated on the path root→node (for B&C)
    cut_lhs: list[np.ndarray] = field(default_factory=list)
    cut_rhs: list[float] = field(default_factory=list)

    # Filled after LP solve
    lp_status: Optional[str] = None
    lp_obj: Optional[float] = None
    lp_x: Optional[np.ndarray] = None
    decision: Optional[str] = None  # "pruned_infeasible", "pruned_bound",
                                    # "pruned_integer", "branched"


@dataclass
class NodeLog:
    """Immutable record written to the execution log."""
    node_id: int
    parent_id: Optional[int]
    depth: int
    branch_var_name: Optional[str]
    branch_dir: Optional[str]
    branch_val: Optional[float]
    lp_status: str
    lp_obj: Optional[float]
    best_incumbent: Optional[float]
    decision: str
    cuts_added: int = 0
