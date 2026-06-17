from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Node:
    # identidade estrutural do nó na árvore de busca;
    # parent_id=None identifica a raiz.
    id: int
    parent_id: Optional[int]
    depth: int

    # informação do branching que gerou este nó a partir do pai;
    # descreve exatamente qual restrição de ramificação foi aplicada.
    branch_var: Optional[int] = None
    branch_dir: Optional[str] = None   # "down" (<=floor) or "up" (>=ceil)
    branch_val: Optional[float] = None

    # bounds locais do nó; nasce como cópia do pai e é apertado com a nova decisão.
    # isso evita mutação global e mantém cada caminho da árvore independente.
    bounds: list[tuple[float | None, float | None]] = field(default_factory=list)

    # cortes extras acumulados no caminho raiz -> nó;
    # no branch-and-cut, filhos herdam esse histórico para manter consistência.
    cut_lhs: list[np.ndarray] = field(default_factory=list)
    cut_rhs: list[float] = field(default_factory=list)

    # preenchidos após resolver a relaxação LP do nó.
    # esses campos registram o estado local antes da decisão de poda/ramificação.
    lp_status: Optional[str] = None
    lp_obj: Optional[float] = None
    lp_x: Optional[np.ndarray] = None
    decision: Optional[str] = None  # "pruned_infeasible", "pruned_bound",
                                    # "pruned_integer", "branched"


@dataclass
class NodeLog:
    """Registro imutável persistido no log de execução."""
    # snapshot do momento em que o nó foi processado;
    # desacopla visualização/auditoria do objeto mutável Node em memória.
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
