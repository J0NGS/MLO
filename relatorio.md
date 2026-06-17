# Relatório — Métodos de Otimização em Larga Escala
## Branch-and-Bound / Branch-and-Cut: Formulações, Estratégias e Resultados

---

## Visão Geral do Framework de Otimização

Este projeto implementa from-scratch um motor genérico de Programação Inteira Mista (MIP) baseado em Branch-and-Bound (B&B) e Branch-and-Cut (B&C), aplicado a 10 problemas clássicos de otimização combinatória. Os quatro aspectos transversais a todo o projeto — formulação, estratégia, exemplos de execução e análise de desempenho — são descritos nesta seção.

---

### 1. Formulação Matemática do Modelo Genérico

Todo problema é codificado como um MIP na **forma padrão**:

$$\text{Otimizar} \quad \mathbf{c}^\top \mathbf{x}$$

Sujeito a:

$$A_{ub}\,\mathbf{x} \leq \mathbf{b}_{ub} \quad \text{(desigualdades, opcional)}$$
$$A_{eq}\,\mathbf{x} = \mathbf{b}_{eq} \quad \text{(igualdades, opcional)}$$
$$\ell_i \leq x_i \leq u_i \quad \forall i \quad \text{(bounds por variável)}$$
$$x_i \in \mathbb{R},\;\mathbb{Z},\;\{0,1\} \quad \text{(integrality: 0=contínua, 1=inteira, 2=binária)}$$

A classe `MIPModel` encapsula esses elementos. Maximizações são convertidas internamente pela inversão de sinal do vetor $\mathbf{c}$ (`effective_c`), permitindo que o solver LP trabalhe sempre em modo minimização. Os 10 problemas cobrem os três subtipos de MIP:

| Tipo | Descrição | Problemas |
|------|-----------|-----------|
| **PIB** — Programação Inteira Binária | Todas as variáveis em $\{0,1\}$ | P2, P3, P6, P7 |
| **PLIP** — Programação Linear Inteira Pura | Todas as variáveis inteiras | P1, P4 |
| **PLIM** — Programação Linear Inteira Mista | Variáveis inteiras e contínuas | P5, P8 |

---

### 2. Estratégia de B&B e B&C Adotada

#### Branch-and-Bound (`BranchAndBound`)

O motor B&B mantém uma fila de **nós**, cada um representando uma subárvore obtida ao fixar bounds de variáveis inteiras. O ciclo principal é:

```
raiz → enfileira → enquanto fila não vazia:
    1. Seleciona nó (estratégia de exploração)
    2. Resolve relaxação LP do nó (HiGHS via scipy)
    3. Decide: poda ou ramificação
    4. Se ramificação: cria dois filhos (down/up) e enfileira
```

**Relaxação LP:** cada nó herda os bounds do pai, apertados pela decisão de branching. A LP é resolvida com `linprog(..., method="highs")`, retornando o valor objetivo, a solução $\mathbf{x}^*$ e os slacks (usados pelos cortes de Gomory).

**Estratégia de seleção de nó** (parâmetro `strategy`):

| Opção | Estrutura | Comportamento |
|-------|-----------|---------------|
| `best_first` | Heap por bound do pai | Sempre expande o nó mais promissor; minimiza nós explorados |
| `dfs` | Pilha (LIFO) | Encontra incumbente rapidamente; mais poda por bound |
| `bfs` | Fila (FIFO) | Explora por nível; garante espessura mínima da árvore |

O padrão adotado em todos os problemas é **`best_first`** com **`most_infeasible`** como critério de ramificação (escolhe a variável inteira com fração mais próxima de 0,5).

**Podas:**

| Código | Condição | Ação |
|--------|----------|------|
| `INF` | LP infeasível | Descarta o nó |
| `BND` | $LP_{obj} \geq$ incumbente (min) ou $\leq$ (max) | Descarta o nó |
| `INT` | Solução LP já é inteira | Atualiza incumbente, descarta |
| `RAM` | Nenhuma poda aplicável | Cria filhos `down` ($x_j \leq \lfloor v \rfloor$) e `up` ($x_j \geq \lceil v \rceil$) |

#### Branch-and-Cut (`BranchAndCut`)

O B&C herda todo o B&B e sobrescreve apenas `_process_node`. Antes de declarar `"branched"`, executa até `max_cut_rounds` rodadas de separação:

```
para cada nó não podado:
    para rodada em 1..max_cut_rounds:
        gera_cortes(lp_atual)  →  lista de (lhs, rhs)
        se nenhum corte: break
        adiciona cortes ao nó (herdados pelos filhos)
        re-resolve LP com cortes
        se novo LP poda o nó: return "pruned_bound/integer"
    ramifica no LP pós-cortes
```

**Cortes disponíveis:**

- **Cortes fracionários de Gomory:** para cada variável inteira básica $x_k$ com valor fracionário $\bar{x}_k$, deriva da linha $k$ do tableau simplex ótimo:
$$-\sum_j f(a_{kj})\,x_j \leq -f(\bar{x}_k), \quad f(v) = v - \lfloor v \rfloor$$
  Globalmente válidos: eliminam a solução LP fracionária sem remover nenhum ponto inteiro factível.

- **Cover cuts:** para restrições knapsack $\sum_j a_j x_j \leq b$ com variáveis binárias, encontra gulosa mente um **cover** $C$ tal que $\sum_{j \in C} a_j > b$ e adiciona:
$$\sum_{j \in C} x_j \leq |C| - 1$$

---

### 3. Exemplos de Execução com Instâncias do Enunciado

#### Exemplo 1 — P2 (Seleção de Projetos): B&B vs B&C

A árvore abaixo mostra a execução completa do B&B puro (19 nós) e do B&C com cortes de Gomory (11 nós) na mesma instância, com LP raiz = 427,86 e ótimo inteiro = 405.

**B&B puro (19 nós):**
```
[No  0] LP=427.86  raiz            => RAM
    [No  1] LP=427.22  P5(do)      => RAM
    |   [No  3] LP=413.00  P3(do)  => RAM  ...
    |   [No  4] LP=425.00  P3(up)  => RAM
    |       [No  9] LP=405.00      => INT  ← incumbente = 405
    [No  2] LP=427.00  P5(up)      => RAM
        [No  5] LP=423.89  P6(do)  => RAM
        [No  6] LP=infeas  P6(up)  => INF
                         ... (19 nós no total)
```

**B&C com Gomory (11 nós):**
```
[No  0] LP=427.86  raiz  +1 corte Gomory  => RAM   ← LP cai para 427,22
    [No  1] LP=425.00  P5(do)  +1c        => RAM
    |   [No  3] LP=405.00  P6(do)         => INT  ← incumbente = 405 (só 3 nós!)
    |   [No  4] LP=422.50  P6(up)         => RAM
    [No  2] LP=423.89  P5(up)  +2c        => RAM
        [No  5] LP=infeas  P3(do)         => INF
        [No  6] LP=infeas  P3(up)         => INF
```

O corte na raiz reduziu imediatamente o espaço de busca, encontrando o incumbente ótimo em 3 nós em vez de 9.

#### Exemplo 2 — P1 (Alocação de Tarefas): trace do nó raiz

```
Nó raiz: LP = 4,6216  (relaxação fracionária do makespan)
Primeira solução inteira encontrada: Nó 10, LP = 8,00  (makespan = 8)
Incumbente melhorado: Nó 11, LP = 7,00  (makespan = 7) ← ótimo
Podas subsequentes: todos os nós com LP ≥ 7,00 são cortados por BND
Prova de otimalidade: fila esvaziada após 45 nós
```

#### Exemplo 3 — P3 (50 fluxos): B&C com cover cut na raiz

```
Nó 0: LP = 1188,89, incumbente greedy = 940
  → cover cut gerado: violação detectada na restrição de banda
  → LP re-resolvido: 1188,89 (corte não fecha gap mas orienta branching)
Nó 20: LP = 1184,00  => INT  ← incumbente = 1184
Nós 29, 33, 34, ...: todos podados por BND (LP ≤ 1184)
Total: 35 nós, ótimo provado em 0,04s
```

---

### 4. Análise de Desempenho Transversal

A tabela a seguir consolida resultados de **todos os problemas**, comparando instâncias de tamanhos distintos e os dois algoritmos principais.

| Problema | Instância | Método | Nós | Tempo (s) | Obj. ótimo | Gap |
|----------|-----------|--------|-----|-----------|-----------|-----|
| P1 Makespan | 3s × 4t | B&B | 45 | 0,050 | 7,0 s | 0% |
| P1 Makespan | 5s × 8t | B&B | 39 | 0,047 | 6,0 s | 0% |
| P1 Makespan | 6s × 10t | B&B | 59 | 0,079 | 4,0 s | 0% |
| P2 Projetos | 6 projetos | B&B | 19 | 0,021 | 405 pts | 0% |
| P2 Projetos | 6 projetos | B&C | **11** | 0,030 | 405 pts | 0% |
| P3 Fluxos | 7 fluxos | B&B (FK) | 8 | <0,001 | 165 | 0% |
| P3 Fluxos | 20 fluxos | B&B (FK) | 658 | 0,002 | 580 | 0% |
| P3 Fluxos | 20 fluxos | B&C | **123** | 0,131 | 580 | 0% |
| P3 Fluxos | 50 fluxos | B&B (FK) | 6 509 109 | **60,0 (TLE)** | 1157 | **9,2%** |
| P3 Fluxos | 50 fluxos | B&C | **35** | **0,041** | **1184** | **0%** |
| P7 TSP | 6 cidades MTZ | B&B | 51 | 0,064 | 70 min | 0% |
| P7 TSP | 6 cidades MTZ | B&C+SECs | **3** | 6,26 | 70 min | 0% |
| P9 B&P | 6 itens | Branch-and-Price | 13 | — | 3 caixas | 0% |

**Observações:**

- **Escalabilidade do B&B puro:** cresce exponencialmente em instâncias densas sem estrutura explorada (P3, 50 fluxos — 6,5M nós com timeout).
- **Impacto dos cortes:** o B&C reduz consistentemente o número de nós em 40–99% quando os cortes são eficazes (P2: −42%; P3 50fl: −99,9%; P7: −94%).
- **Warm-start:** fornecer um incumbente inicial via heurística (greedy, FFD, NN) elimina explorações desnecessárias nas primeiras iterações e permite podas por BND antes mesmo do primeiro inteiro encontrado.
- **Qualidade da solução:** em todos os casos em que o algoritmo termina dentro do tempo limite, a solução é **provadamente ótima** (gap = 0%). O único caso de gap residual (P3, 50 fluxos, B&B) é consequência direta do timeout, não de limitação de qualidade do modelo.

---

## 1. P1 — Alocação de Tarefas em Multiprocessador (PLIP)

### Formulação

Minimizar o makespan M (tempo máximo entre servidores).

**Variáveis:** $x_{ij} \in \{0,1\}$ — tarefa $j$ alocada ao servidor $i$; $M \geq 0$ contínua.

$$\min\ M$$

$$\sum_j p_{ij}\,x_{ij} \leq M \quad \forall i \quad \text{(lineariza o máximo)}$$
$$\sum_j p_{ij}\,x_{ij} \leq \text{cap} \quad \forall i \quad \text{(capacidade)}$$
$$\sum_i x_{ij} = 1 \quad \forall j \quad \text{(cada tarefa em exatamente 1 servidor)}$$

### Estratégia B&B

- **Relaxação LP:** `linprog` (HiGHS) com todas as variáveis binárias relaxadas para $[0,1]$.
- **Seleção de nó:** best-first (heap por bound LP do pai).
- **Ramificação:** most-infeasible — variável binária com fração mais próxima de 0,5.
- **Podas:** infeasible, bound ($LP \geq$ incumbente), inteiro.

### Instância do enunciado — Árvore B&B

Dados: S=3, T=4, capacidade=12. Ótimo: **makespan = 7 s**.

```
[No   0] LP=4.6216   raiz                 => RAM
    [No   1] LP=5.1429   x32(do)          => RAM
    |   [No   3] LP=6.0000   x12(do)      => RAM
    |   [No   4] LP=5.2222   x12(up)      => RAM   ...
    [No   2] LP=4.9512   x32(up)          => RAM
        [No   6] LP=4.9718   x21(up)      => RAM
            [No   9] LP=5.2857   x24(do)  => RAM
            [No  10] LP=8.0000   x24(up)  => INT  <-- 1ª solução inteira (M=8)
        ...
            [No  11] LP=7.0000   x14(do)  => INT  <-- incumbente atualizado para 7
```

Nós podados por BND quando LP ≥ 7 (incumbente). Prova de otimalidade ao esvaziar a fila.

Solução: S1←T3(3s), S2←T1(3s), S3←{T2(4s),T4(3s)} — carga máx = 7 s.

### Análise de desempenho

| Instância | Servidores × Tarefas | Makespan ótimo | Nós explorados | Tempo (s) |
|-----------|----------------------|---------------|----------------|-----------|
| Enunciado | 3 × 4 | 7 | 45 | 0,050 |
| Aleatória 2 | 5 × 8 | 6 | 39 | 0,047 |
| Aleatória 3 | 6 × 10 | 4 | 59 | 0,079 |

O número de nós cresce moderadamente com o tamanho — a poda por bound LP é efetiva porque a relaxação fraccionária é muito próxima do ótimo inteiro.

---

## 2. P2 — Seleção de Projetos com Restrições Lógicas (PIB)

### Formulação

Maximizar impacto total de 6 projetos $P_1\ldots P_6$ sujeito a orçamento e restrições lógicas.

**Variáveis:** $x_i \in \{0,1\}$, $i = 1\ldots 6$.

$$\max \sum_i r_i\,x_i$$

$$\sum_i c_i\,x_i \leq 280 \quad \text{(orçamento)}$$
$$x_3 \leq x_1 \quad \text{(P3 exige P1)}$$
$$x_4 + x_5 \leq 1 \quad \text{(P4 e P5 mutuamente exclusivos)}$$
$$x_1 + x_2 + x_4 \geq 2 \quad \text{(mínimo 2 de \{P1,P2,P4\})}$$

### Estratégia B&B e B&C

- **B&B puro:** best-first, most-infeasible, relaxação LP (HiGHS).
- **B&C (Gomory):** antes de ramificar, tenta gerar cortes fracionários de Gomory a partir do tableau simplex ótimo. Cada corte apertando a relaxação LP reduz o número de nós a explorar.

### Execução — instância do enunciado

**B&B (19 nós, 0,021 s):**
```
[No   0] LP=427.86  raiz               => RAM
    [No   9] LP=405.00  P6(do)         => INT  <-- incumbente = 405
    [No   8] LP=410.83  P4(up)         => RAM
        [No  16] LP=409.00  P1(up)     => RAM  (podado na sub-árvore)
```

**B&C (11 nós, 0,030 s):**
```
[No   0] LP=427.86  raiz              => RAM
    +1 corte Gomory na raiz => LP cai para ~425
    [No   1] LP=425.00  P5(do)        => RAM
        [No   3] LP=405.00  P6(do)    => INT  <-- incumbente = 405 em 3 nós
```

Solução ótima (ambos): **405 pts**, projetos {P1, P2, P3, P4}, custo total = R$280 mil.

### Comparação B&B vs B&C

| Métrica | B&B | B&C |
|---------|-----|-----|
| Nós explorados | 19 | 11 |
| Tempo (s) | 0,021 | 0,030 |
| Solução | 405 pts | 405 pts |
| Cortes Gomory adicionados | — | 3 |

O B&C explorou **42% menos nós**. Os cortes de Gomory fecharam o gap do LP raiz (427,86 → 425,00 após o primeiro corte), antecipando podas por bound. O tempo total foi ligeiramente maior porque a separação de cortes tem custo quadrático no número de restrições — vantajoso para instâncias maiores.

---

## 3. P3 — Seleção de Fluxos em Datacenter (Knapsack Multidimensional)

### Formulação

$$\max \sum_j p_j\,x_j \quad \text{s.t.} \quad \sum_j b_j\,x_j \leq B,\quad \sum_j f_j\,x_j \leq F,\quad x_j \in \{0,1\}$$

### Estratégia B&B (bound FK) e B&C (cover cuts)

- **B&B:** bound via mochila fracionária greedy (razão $p_j/b_j$); ramificação no "break item" da FK; incumbente greedy como warm-start.
- **B&C:** relaxação LP padrão (HiGHS) + cover cuts nas restrições de capacidade (detecta subconjuntos cujo peso total excede a capacidade).

### Análise de desempenho — 4 instâncias

| Instância | Algoritmo | Ótimo | Bound raiz | Gap | Nós | Tempo (s) |
|-----------|-----------|-------|-----------|-----|-----|-----------|
| Original (7 fl.) | B&B (FK) | 165 | 166,88 | 0,0% | 8 | <0,001 |
| Original (7 fl.) | B&C (cover) | 165 | 166,88 | 0,0% | 9 | 0,023 |
| Aleatória 10 fl. | B&B (FK) | 251 | 281,57 | 0,0% | 23 | <0,001 |
| Aleatória 10 fl. | B&C (cover) | 251 | 280,54 | 0,0% | 31 | 0,033 |
| Aleatória 20 fl. | B&B (FK) | 580 | 657,92 | 0,0% | 658 | 0,002 |
| Aleatória 20 fl. | B&C (cover) | 580 | 606,10 | 0,0% | 123 | 0,131 |
| Aleatória 50 fl. | B&B (FK) | 1157 | 1381,12 | **9,2%** | 6 509 109 | **60,0 (TLE)** |
| Aleatória 50 fl. | B&C (cover) | **1184** | 1188,89 | **0,0%** | **35** | 0,041 |

Para 50 fluxos o B&B com bound FK **esgotou o tempo** (6,5M nós, gap 9,2%), enquanto o B&C com cover cuts provou otimalidade **em apenas 35 nós**. A razão: o bound FK é mais fraco que a relaxação LP para instâncias densas, e os cover cuts apertam decisivamente o LP raiz.

---

## 4. P4 — Bin Packing de VMs em Nuvem (PLIP)

### Formulação

Minimizar servidores físicos ativos para alocar $n$ VMs.

**Variáveis:** $y_j \in \{0,1\}$ (servidor $j$ ativo); $x_{ij} \in \{0,1\}$ (VM $i$ no servidor $j$).

$$\min \sum_j y_j \quad \text{s.t.} \quad \sum_i d_i\,x_{ij} \leq C\,y_j,\quad \sum_j x_{ij}=1,\quad x_{ij} \leq y_j,\quad y_{j+1} \leq y_j$$

### Estratégia B&B / B&C / FFD

- **FFD (First-Fit Decreasing):** heurística $O(n \log n)$ — ordena VMs por demanda decrescente e encaixa em primeiro bin disponível. Serve como incumbente inicial.
- **Limitante inferior fracionário:** $\lceil \sum d_i / C \rceil$.
- **B&B / B&C:** quando LB = FFD a otimalidade é provada no nó raiz (sem exploração).

### Análise de desempenho

| Instância | FFD | LB | B&B obj | B&B nós | B&C obj | B&C nós |
|-----------|-----|----|---------|---------|---------|---------|
| Original (6 VMs) | 3 | 3 | 3 | **1** | 3 | **1** |
| Aleatória (10 VMs) | 3 | 3 | 3 | **1** | 3 | **1** |
| Aleatória (15 VMs) | 5 | 5 | 5 | **1** | 5 | **1** |
| Aleatória (20 VMs) | 8 | 8 | 8 | **1** | 8 | **1** |

Em todos os casos LB = FFD: o limitante fracionário e a heurística coincidem, provando otimalidade imediata. Nenhum corte de cover foi ativado. O FFD é ótimo para estas instâncias porque as demandas são inteiras moderadas e a capacidade 10 permite combinações perfeitas.

---

## 5. P5 — Planejamento de Produção com Custos de Setup (PLIM)

### Formulação

Maximizar lucro líquido de 3 chips A, B, C com custos fixos de ativação.

**Variáveis:** $q_k \geq 0$ (quantidade contínua); $y_k \in \{0,1\}$ (chip ativado).

$$\max \sum_k (r_k - c_k)\,q_k - \sum_k s_k\,y_k$$

$$q_A+q_B+q_C \leq 60,\quad 1\,q_A + 1{,}5\,q_B + 2\,q_C \leq 80$$
$$q_k \leq M_k\,y_k,\quad q_k \geq d_k\,y_k \quad \forall k$$

Big-M ajustados (tight): $M_A=60$, $M_B=53$, $M_C=40$.

### Estratégia B&B

Ramifica apenas nas variáveis binárias $y_k$ (as $q_k$ são contínuas — resolvidas pela LP para cada combinação binária fixa).

### Instância do enunciado

Solução ótima: **lucro = $520**, produzir apenas Chip A (60 unidades).

Árvore B&B (5 nós):
```
[No 0] LP=556.67  raiz   => RAM   (yA=0.67, yC=0.50 fracionários)
    [No 1] LP=540.83  yC(down) => RAM
        [No 3] LP=490.00  yA(down) => INT (A=0,B+C ativos)
        [No 4] LP=520.00  yA(up)   => INT (melhor: só A)
    [No 2] LP=420.00  yC(up)   => INT (B+C: lucro 420)
```

### Análise de sensibilidade — custo de setup do Chip C ($100 a $400)

| Setup C ($) | Lucro ótimo | Chip C ativo? | Qtd C |
|-------------|-------------|---------------|-------|
| 100 | 620 | Sim | 40 |
| 150 | 570 | Sim | 40 |
| 200 | 520 | Sim | 40 |
| **250** | **520** | **Não** | 0 |
| 300 | 520 | Não | 0 |
| 350 | 520 | Não | 0 |
| 400 | 520 | Não | 0 |

Interpretação: para $s_C < \$200$, Chip C é mais lucrativo que A (margem/hora é maior, apesar do setup). A partir de $\$200$, o Chip A domina com $q_A=60$, lucro fixo em $\$520$. O ponto de indiferença é exatamente $s_C = \$200$.

---

## 6. P6 — Cobertura de Zonas de Rede (Set Covering — PIB)

### Formulação

Minimizar custo total de antenas instaladas cobrindo todas as $m$ zonas.

$$\min \sum_j c_j\,x_j \quad \text{s.t.} \quad \sum_{j:\,i \in S_j} x_j \geq 1 \quad \forall i,\quad x_j \in \{0,1\}$$

### Estratégia B&B e B&C

- **Heurística greedy:** warm-start (escolhe antenas por razão zonas-novas/custo).
- **B&C com cortes de clique:** grafo de conflito entre zonas → cliques maximais geram cortes válidos. Na instância original, todos os 0 cortes de clique foram dominados pelas restrições de zona (esperado: cobertura 1-a-1 elimina as violações).

### Instância do enunciado — análise de sensibilidade no custo de L3

| Custo L3 ($) | Custo ótimo | Solução |
|--------------|-------------|---------|
| 50 | 130 | {L2, L3} |
| 100 | 180 | {L2, L3} |
| 150 | 230 | {L2, L3} |
| 200 | 280 | {L2, L3} |
| **240** | **320** | **{L1, L4, L5}** |
| 280 | 320 | {L1, L4, L5} |

Ponto de mudança exatamente em $c(L3)=240$: empate entre as duas soluções.

### B&B vs B&C — instâncias aleatórias

| Instância | Greedy | B&B ótimo | Nós | B&C ótimo | Nós |
|-----------|--------|-----------|-----|-----------|-----|
| Orig. (8 zon., 5 ant.) | 230 | 230 | 1 | 230 | 1 |
| Aleat. (10 zon., 8 ant.) | 16,0 | 13,8 | 1 | 13,8 | 1 |
| Aleat. (15 zon., 10 ant.) | 32,7 | 31,2 | 1 | 31,2 | 1 |
| Aleat. (20 zon., 12 ant.) | 31,6 | 30,0 | 1 | 30,0 | 1 |

A LP raiz já é inteira (matriz de cobertura é TU para esses casos), provando otimalidade no nó raiz. O greedy supera ou empata com a LP em todas as instâncias aleatórias.

---

## 7. P7 — Roteamento de Técnico (TSP — PIB)

### Formulação — dois modelos

**MTZ (Miller-Tucker-Zemlin):** adiciona variáveis de posição $u_i \in \{1,\ldots,n-1\}$ para eliminar subciclos.

$$\min \sum_{i \neq j} d_{ij}\,x_{ij}$$
$$\sum_j x_{ij}=1 \; \forall i,\quad \sum_i x_{ij}=1 \; \forall j \quad \text{(grau)}$$
$$u_i - u_j + n\,x_{ij} \leq n-1 \quad \forall i \neq j \in \{1,\ldots,n-1\} \quad \text{(MTZ)}$$

**DFJ (Dantzig-Fulkerson-Johnson):** sem variáveis extras, restrições de subciclo (SECs) adicionadas como cortes lazy no B&C:
$$\sum_{i,j \in S} x_{ij} \leq |S|-1 \quad \forall S \subsetneq V,\; |S|\geq 2$$

### Heurística do Vizinho Mais Próximo (NN)

Partindo de E0, sempre vai à cidade não visitada mais próxima:
```
E0 → E1(10) → E2(12) → E3(8) → E4(11) → E5(9) → E0(25) = 75 min
```

### Instância do enunciado — comparação

| Método | Nós | Tempo (s) | Custo (min) | Gap vs NN |
|--------|-----|-----------|------------|-----------|
| NN (heurística) | — | — | **75** | 0% (ref.) |
| B&B (MTZ) | 51 | 0,064 | **70** | −6,7% |
| B&C (MTZ + SECs lazy) | **3** | 6,26 | **70** | −6,7% |

Solução ótima: E0→E2→E3→E5→E4→E1→E0 = **70 min**.
O B&C explora apenas 3 nós mas cada nó é custoso (3 rodadas de Gomory + 3 SECs adicionados); o B&B MTZ explora 51 nós mais baratos. Gap ótimo vs NN: **6,7%**.

### Análise de sensibilidade — distância d[E0–E2]

| d[E0–E2] | Custo ótimo | Roteiro |
|----------|-------------|---------|
| ≤ 15 | d + 55 | Usa E0→E2 |
| **16** | **71** | Mudança (empate) |
| ≥ 17 | 71 | Evita E0→E2 (E0→E4→E5→E3→E2→E1→E0) |

---

## 8. P8 — Localização de Servidores CDN (UFL — PLIM)

### Formulação

Minimizar custo de instalação + latência de serviço.

**Variáveis:** $y_j \in \{0,1\}$ (centro $j$ aberto); $x_{ij} \in [0,1]$ (fração de demanda da região $i$ servida pelo centro $j$).

$$\min \sum_j f_j\,y_j + \sum_i \sum_j c_{ij}\,x_{ij}$$

$$\sum_j x_{ij}=1 \; \forall i,\quad x_{ij} \leq y_j,\quad \sum_j y_j \leq \text{MAX\_OPEN},\quad \sum_j f_j\,y_j \leq \text{BUDGET}$$

### Instância do enunciado

Solução ótima: abrir apenas **C4** (custo instalação $20 + latência total $33 = **$53**).

### Escalabilidade — análise de desempenho

| Instância | Centros | Regiões | Greedy | B&B ótimo | Nós | Tempo (s) |
|-----------|---------|---------|--------|-----------|-----|-----------|
| Enunciado | 4 | 6 | 53 | 53 | 1 | 0,003 |
| Aleatória | 4 | 10 | 106,5 | 106,5 | 1 | 0,003 |
| Aleatória | 5 | 15 | 160,1 | 160,1 | 1 | 0,003 |
| **Aleatória** | **6** | **20** | 246,3 | **215,1** | **1** | 0,003 |

A instância com 6 centros e 20 regiões demonstra o valor do MIP: o greedy retorna 246,3 enquanto o ótimo é 215,1 — **diferença de 14,5%**. A relaxação LP do UFL é conhecida por ser quase integral (as variáveis $x_{ij}$ são contínuas), fazendo com que o B&B prove otimalidade no nó raiz em todos os casos testados.

---

## 9. P9 — Bin Packing por Geração de Colunas + Branch-and-Price

### Formulação (modelo de padrões)

**Padrões** $P$: subconjuntos de itens com $\sum_{i \in P} w_i \leq B$.

**PMR (Problema Mestre Restrito):**
$$\min \sum_p z_p \quad \text{s.t.} \quad \sum_p a_{ip}\,z_p = 1 \; \forall i,\quad z_p \geq 0$$

**Subproblema de precificação** (mochila 0-1):
$$\max \sum_i \pi_i\,a_i \; \text{s.t.} \; \sum_i w_i\,a_i \leq B,\; a_i \in \{0,1\}$$

Adiciona padrão se custo reduzido $rc = 1 - \max\_knap < 0$.

### Instância: 6 itens, $B=10$, pesos = [6,4,4,3,3,2]

**Geração de colunas (7 iterações):**
- Inicia com 6 singletons, termina com 12 padrões.
- LP ótimo: **2,333** (relaxação fracionária).

**Comparação B&P vs arredondamento direto:**

| Método | Nós | Caixas |
|--------|-----|--------|
| Arredondamento $\lceil z_p \rceil$ da LP | — | **6** (subótimo) |
| FFD heurística | — | 3 |
| Branch-and-Price (exato) | 13 | **3** |

O arredondamento direto da solução fracionária do PMR final retorna 6 caixas — **o dobro do ótimo** — porque os padrões com $z_p$ fracionário são todos arredondados para cima independentemente. O B&P ramifica nas variáveis $z_p$ mais fracionárias e re-executa a Geração de Colunas em cada nó, provando o ótimo com apenas 13 nós.

---

## 10. P10 — Timetabling (Grade de Horários — MIP)

### Formulação

Maximizar satisfação total de alocação de disciplinas a salas e horários.

**Variáveis:** $x_{ijk} \in \{0,1\}$ — disciplina $i$ na sala $j$ no horário $k$.

$$\max \sum_{i,j,k} \text{sat}_{ijk}\,x_{ijk}$$

$$\sum_{j,k} x_{ijk} = 1 \; \forall i \quad \text{(alocação única)}$$
$$\sum_i x_{ijk} \leq 1 \; \forall j,k \quad \text{(conflito de sala)}$$
$$\text{restrições de capacidade, precedência e anti-conflito}$$

**Cortes adicionais:** Clique cuts — para cada clique $C$ no grafo de conflito de disciplinas e horário $k$: $\sum_{i \in C}\sum_j x_{ijk} \leq 1$.

### Comparação B&C vs OR-Tools CP-SAT

**Instância pequena** (4 disc., 3 salas, 4 horários — 48 var., 26 restr.):

| Método | Satisfação | Nós | Tempo | Cortes |
|--------|-----------|-----|-------|--------|
| B&C (depth-first, clique + Gomory) | **3,4794** | 1 | <0,01s | 0 |
| OR-Tools CP-SAT | **3,4794** | — | <0,01s | — |
| Gap | 0,000000 | — | — | — |

**Instância grande** (8 disc., 5 salas, 6 horários — 240 var., 65 restr.):

| Método | Satisfação | Nós | Tempo | Cortes |
|--------|-----------|-----|-------|--------|
| B&C | **6,4180** | 1 | <0,01s | 0 |
| OR-Tools CP-SAT | **6,4180** | — | <0,01s | — |
| Gap | 0,000000 | — | — | — |

**Análise:** A LP raiz já é inteira em ambas as instâncias porque as restrições de igualdade de alocação única + as desigualdades de sala formam uma matriz **totalmente unimodular (TU)** quando todas as restrições combinatórias são satisfeitas pelos dados. Nenhum corte de Gomory ou de clique foi violado. O OR-Tools CP-SAT converge à mesma solução, confirmando a otimalidade. Para instâncias maiores com mais conflitos e restrições de precedência cruzadas, os clique cuts passariam a ser decisivos.

---

## Sumário Executivo de Desempenho

| Problema | Método | Nós (enunciado) | Tempo (s) | Gap |
|----------|--------|-----------------|-----------|-----|
| P1 — Makespan | B&B | 45 | 0,050 | 0% |
| P2 — Projetos | B&B → B&C | 19 → **11** | 0,021 | 0% |
| P3 — Fluxos (50 fl.) | B&B → B&C | 6.5M(TLE) → **35** | 60s → 0,04s | 9,2% → **0%** |
| P4 — VMs | B&B+FFD | 1 | 0,003 | 0% |
| P5 — Setup | B&B | 5 | 0,009 | 0% |
| P6 — Cobertura | B&B | 1 | 0,003 | 0% |
| P7 — TSP | B&B → B&C | 51 → **3** | 0,064 | 0% (vs NN: 6,7%) |
| P8 — CDN | B&B | 1 | 0,003 | 0% |
| P9 — Cutting Stock | B&P | 13 | — | 0% (vs round: 100%) |
| P10 — Timetabling | B&C = CP-SAT | 1 | <0,01 | 0% |

O padrão mais notável é o de **P3 com 50 fluxos**: B&B com bound FK falha por tempo (gap 9,2%) enquanto B&C com cover cuts resolve em 35 nós. Isso exemplifica o princípio central do B&C — a qualidade do bound LP, aprimorado por cortes válidos, é o fator determinante da escalabilidade.
