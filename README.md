# MLO — Solvers de Programação Inteira Mista (MIP)

Implementação from-scratch de Branch-and-Bound (B&B) e Branch-and-Cut (B&C) em Python puro, aplicada a 10 problemas clássicos de otimização combinatória.

---

## Estrutura do projeto

```
MLO/
├── src/
│   ├── core/               # Motor genérico de otimização
│   │   ├── model.py        # Representação matemática do MIP
│   │   ├── node.py         # Nó da árvore B&B + log imutável
│   │   ├── relaxation.py   # Resolve a relaxação LP de cada nó (HiGHS)
│   │   ├── branch_bound.py # Engine de Branch-and-Bound genérico
│   │   ├── branch_cut.py   # Engine de Branch-and-Cut (estende B&B)
│   │   └── cuts.py         # Geradores de cortes (Gomory, Cover)
│   │
│   └── problems/           # Um arquivo por problema aplicado
│       ├── p1_task_allocation.py
│       ├── p2_project_selection.py
│       ├── p3_datacenter_flows.py
│       ├── p4_vm_binpacking.py
│       ├── p5_production_setup.py
│       ├── p6_network_coverage.py
│       ├── p7_tsp.py
│       ├── p8_facility_location.py
│       ├── p9_bin_packing.py
│       └── p10_timetabling.py
└── Lista_MMOL.pdf          # Lista de exercícios de referência
```

---

## Dependências

```
python >= 3.10
numpy
scipy      # linprog / HiGHS como solver LP interno
ortools    # opcional — comparação em p10_timetabling.py
```

---

## Core — Como funciona

### `MIPModel`

Representa qualquer MIP na forma padrão:

```
Otimizar (min/max):  c @ x
Sujeito a:
    A_ub @ x <= b_ub   (desigualdades, opcional)
    A_eq @ x == b_eq   (igualdades, opcional)
    x_i ∈ [lb_i, ub_i]
    x_i ∈ {contínua | inteira | binária}
```

```python
from core import MIPModel
import numpy as np

model = MIPModel(
    c          = [1.0, 2.0],
    A_ub       = [[1, 1]],
    b_ub       = [10.0],
    bounds     = [(0, None), (0, None)],
    integrality= [1, 1],        # 0=contínua  1=inteira  2=binária
    sense      = "max",
    var_names  = ["x1", "x2"],
)
```

---

### `BranchAndBound`

Engine genérico de B&B. Resolve a relaxação LP de cada nó via HiGHS e aplica as podas clássicas:

| Código | Decisão | Significado |
|--------|---------|-------------|
| `INT`  | pruned_integer    | LP já é inteiro → atualiza incumbente |
| `BND`  | pruned_bound      | Bound LP pior que incumbente → poda |
| `INF`  | pruned_infeasible | Relaxação inviável → poda             |
| `RAM`  | branched          | Cria dois filhos (floor / ceil)        |

**Parâmetros configuráveis:**

| Parâmetro | Opções | Padrão |
|-----------|--------|--------|
| `strategy` | `"best_first"` · `"dfs"` · `"bfs"` | `"best_first"` |
| `branching` | `"most_infeasible"` · `"first_fractional"` | `"most_infeasible"` |
| `time_limit` | float (segundos) | sem limite |
| `initial_incumbent` | float | — (warm start) |

```python
from core import BranchAndBound

solver = BranchAndBound(model, strategy="best_first", branching="most_infeasible")
result = solver.solve()

print(result.status)          # "optimal" | "infeasible" | "timeout"
print(result.obj_value)       # valor ótimo
print(result.x)               # vetor de solução
print(result.nodes_explored)  # nós processados
```

---

### `BranchAndCut`

Estende `BranchAndBound`. Antes de ramificar cada nó, executa rodadas de separação de cortes para apertar o bound LP.

**Tipos de corte disponíveis:**

| Corte | Flag | Aplicação |
|-------|------|-----------|
| Gomory fracionário | `"gomory"` | Qualquer MIP com restrições `<=` |
| Cover cut | `"cover"` | Restrições tipo knapsack binário |

```python
from core import BranchAndCut

solver = BranchAndCut(
    model,
    strategy        = "best_first",
    branching       = "most_infeasible",
    cut_types       = ["gomory", "cover"],
    max_cut_rounds  = 3,
)
result = solver.solve()
```

O B&C tende a explorar menos nós que o B&B puro quando os cortes são efetivos — o tradeoff é o custo de separação por nó.

---

### Saída no terminal (verbose)

```
[No   0] LP=  120.000  inc=---         raiz      =>  RAM
    [No   1] LP=  118.500  inc=---    P3(dw)    =>  RAM
        [No   3] LP=  115.000  inc=115.0  P1(dw)  =>  INT
        [No   4] LP=  114.000  inc=115.0  P1(up)  =>  BND
    [No   2] LP=  116.000  inc=115.0  P3(up)  =>  BND

============================================================
  ARVORE B&B  (5 nos  otimo=115.0000)
============================================================
[No   0] LP= 120.0000  inc=---   raiz  RAM
├── [No   1] LP= 118.5000  inc=---   P3(dw)  RAM
│   ├── [No   3] LP= 115.0000  inc=115.0000  P1(dw)  INT
│   └── [No   4] LP= 114.0000  inc=115.0000  P1(up)  BND
└── [No   2] LP= 116.0000  inc=115.0000  P3(up)  BND
```

---

## Problemas implementados

| # | Arquivo | Tipo | Objetivo | Técnica |
|---|---------|------|----------|---------|
| 1 | `p1_task_allocation.py`   | PLIP | Minimizar makespan em S servidores com T tarefas | B&B |
| 2 | `p2_project_selection.py` | PIB  | Maximizar impacto de projetos sob restrições lógicas e orçamento | B&B + B&C |
| 3 | `p3_datacenter_flows.py`  | PIB  | Knapsack multidimensional — maximizar prioridade de fluxos | B&B + B&C |
| 4 | `p4_vm_binpacking.py`     | PLIP | Bin packing — minimizar servidores físicos para alocar VMs | B&B + B&C |
| 5 | `p5_production_setup.py`  | PLIM | Planejamento de produção com custos fixos de setup (Big-M) | B&B + B&C |
| 6 | `p6_network_coverage.py`  | PIB  | Set covering — mínimo de antenas para cobrir todas as zonas | B&B + B&C |
| 7 | `p7_tsp.py`               | PIB  | TSP (caixeiro-viajante) com formulações DFJ e MTZ | B&C (SECs lazy) |
| 8 | `p8_facility_location.py` | PLIM | Localização de CDN (UFL) — minimizar instalação + latência | B&B + B&C |
| 9 | `p9_bin_packing.py`       | PIB  | Bin packing via Geração de Colunas + Branch-and-Price | CG + B&P |
| 10 | `p10_timetabling.py`     | MIP  | Grade de horários com clique cuts e cortes de Gomory | B&C + OR-Tools |

**Legendas de tipo:** PIB = Programação Inteira Binária · PLIP = PL Inteiro Puro · PLIM = PL Inteiro Misto

---

## Executando um problema

```bash
cd src
python problems/p1_task_allocation.py
python problems/p2_project_selection.py
# ... etc
```

Cada arquivo é autocontido (`if __name__ == "__main__"`) e já inclui instâncias de exemplo prontas para rodar.

---

## Exemplo mínimo

```python
import sys, os
sys.path.insert(0, "src")

from core import MIPModel, BranchAndBound, BranchAndCut

# Maximizar 5x1 + 4x2  sujeito a 6x1+4x2<=24, x1+2x2<=6, xi inteiro
model = MIPModel(
    c           = [5.0, 4.0],
    A_ub        = [[6, 4], [1, 2]],
    b_ub        = [24.0, 6.0],
    bounds      = [(0, None), (0, None)],
    integrality = [1, 1],
    sense       = "max",
    var_names   = ["x1", "x2"],
)

result = BranchAndBound(model).solve()
print(f"Ótimo: {result.obj_value}  x={result.x}")

# Com cortes de Gomory
result_bc = BranchAndCut(model, cut_types=["gomory"]).solve()
print(f"B&C nós: {result_bc.nodes_explored}  B&B nós: {result.nodes_explored}")
```

---

## Arquitetura — decisões de design

- **Separação modelo / engine / problema:** `MIPModel` descreve *o quê* otimizar; `BranchAndBound`/`BranchAndCut` decidem *como* buscar; cada arquivo em `problems/` apenas constrói o modelo e interpreta a solução.
- **`BranchAndCut` herda `BranchAndBound`:** a única diferença real está no método `_process_node`, que tenta cortes antes de declarar `"branched"`.
- **Cortes acumulam no caminho:** cada `Node` carrega `cut_lhs`/`cut_rhs` herdados do pai, garantindo que cortes gerados num ancestral sejam respeitados por todos os descendentes.
- **Relaxação via HiGHS:** `scipy.optimize.linprog(..., method="highs")` — rápido e estável para os tamanhos de instância destes problemas.
