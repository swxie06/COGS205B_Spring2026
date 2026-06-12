## Research question

Can communication pressure alone drive agents to converge on a shared, emergent label system? How does network centralization affect convergence speed and label consistency? Does the listener's update strategy change these dynamics?

## Hypotheses

- H1: Higher network centralization accelerates convergence to a consistent label system
- H2: In highly centralized networks the emergent prototypes are biased toward the hub's initial prototypes (more generally, the emergent prototypes are biased toward the initial prototypes of the agents with higher degrees in the network)
- H3: Low-centralization networks may produce "dialects" - spatially distinct clusters that stabilize on different prototypes
- H4: Unconditional updating converges faster than selective updating; the tradeoff is reduced sensitivity to conflicting information

---

## Stimulus space

- Two continuous features: `color`$\in [0,1]$, `antenna` $\in [0,1]$
- Stimuli sampled uniformly from $[0,1]^2$
- No ground-truth category boundary

## Labels

Two labels: `Dax` and `Leca`

---

## Agent model

Each agent maintains two prototype vectors (one per label), initialized by sampling each coordinate independently from Uniform(0, 1).

Labeling rule: assign the label whose prototype is nearest to the stimulus by Euclidean distance.

---

## Communication protocol

One communication event:

1. Draw a (speaker, listener) pair from neighboring agents according to the network
2. Speaker samples stimulus X uniformly from $[0,1]^2$
3. Speaker computes its label for X
4. Speaker sends (X, label) to listener
5. Listener decides whether to update (see Update strategies below)
6. If updating: $\text{prototype}_{label} \leftarrow (1-\eta) \text{ prototype}_{label} + \eta X$

One timestep -- n communication events.

---

## Update strategies

The listener's update decision is controlled by the `update_rule` parameter:

| Value           | Name             | Behavior                                                                                                                                                    |
| --------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unconditional` | Unconditional    | Listener always updates, regardless of its own classification                                                                                               |
| `conflict`      | Conflict-driven  | Listener updates only if the speaker's label differs from the listener's own label for X                                                                    |
| `ambiguous`     | Ambiguity-driven | Listener updates only if its own classification is ambiguous — i.e., when the distances to the two prototypes differ by less than a tolerance threshold `δ` |

---

## Network structure

Network centralization is treated as a **continuous variable** measured by degree centralization:

$$C = \frac{\sum(d_{max} − d_i)}{(n − 1)(n − 2)}$$

where $d_i$ is the degree of node i and $d_{max}$ is the maximum degree in the graph. C = 1.0 for a star graph; C ≈ 0 for a lattice.

Networks are generated using the **Barabasi-Albert (BA) preferential attachment model**: `G = barabasi_albert_graph(n, m)` where `m` is the number of edges each new node attaches to existing nodes. Smaller m produces fewer, larger hubs and higher C.

Each simulation records the **actual** C of the generated graph (not just the p value). A lattice grid graph and a star graph are included as boundary conditions at C ≈ 0 and C = 1.0 respectively.

---

## Parameters

| Parameter           | Symbol   | Default                            |
| ------------------- | -------- | ---------------------------------- |
| Number of agents    | n        | 100                                |
| Timesteps           | T        | 1000                               |
| Learning rate       | $\eta$   | 0.1                                |
| Network structures  | m        | lattice, $m \in {1,2,5,10}$, star  |
| Update strategies   |          | unconditional, conflict, ambiguous |
| Ambiguity threshold | $\delta$ | 0.1                                |
| Seeds per condition |          | 20                                 |

---

## Measurements

Recorded every 20 timesteps for each (network condition, update rule, seed) triplet.

**Pairwise consistency:** For each of 50 fixed test stimuli, draw 100 random agent pairs and compute the proportion that assign the same label. Average across stimuli and pairs.

**Overall consistency:** Across all agents and all 50 test stimuli, find the majority label per stimulus, then compute the proportion of agent-stimulus pairs that agree with the majority. Average across stimuli.

At the end of each run, also record:

- The actual degree centralization C of the network
- The hub agent's prototype vectors (to test H2)
- Per-agent label prototypes (to test H3)

---

## Hypothesis Evaluation

H1: Centralization Accelerates Convergence

- Plot consistency trajectories across all centralization levels. Generate two plots for each update strategy: pairwise consistency and overall consistency.
- Record convergence time as the number of timesteps required to reach 70% pairwise consistency and 70% overall consistency. If the threshold is not reached, do not exclude the condition; instead, indicate this explicitly in the visualization (e.g., using a faded bar).
- Plot convergence time vs. network centralization.

H2: Hub Bias

- Analyze for unconditional update strategy only.
- Compute the final population prototypes by averaging agent prototypes for each category.
- Compute the distance between the final population prototypes and each agent's initial prototypes.
- Plot this distance vs. agent degree.

H3: Dialects

- Compute prototype diversity as the standard deviation of prototype coordinates for each category.
- Plot prototype diversity trajectories across all centralization levels, with one plot for each update strategy.
- Compute pairwise distances between agents' prototype sets.
- Visualize the resulting distance matrix or agent similarity network for representative runs.
- Examine whether clusters of agents emerge that share similar prototype configurations.

H4: Update Strategies

- For a representative centralization level, record convergence time, final pairwise consistency, final overall consistency, and final prototype diversity for each update strategy.
- Generate one bar plot for each metric.
- For convergence time, if the threshold is not reached, do not exclude the condition; instead, indicate this explicitly in the visualization (e.g., using a faded bar).

---

## Verification requirements

Implemented in `tests/verify.py`:

- **Invariant:** All prototype coordinates remain within $[0,1]$ at every timestep
- **Deterministic seed:** Same seed always produces identical results
- **No-learning ablation:** $\eta=0$ → pairwise consistency ≈ 0.5 throughout
- **Ambiguity ablation:** `update_rule=ambiguous` with δ = 0 → no updates ever occur (listener is never ambiguous when the decision boundary is exact)
- **Edge cases:** n = 2 with `unconditional` → both agents converge to identical prototypes
- **Sensitivity:** Qualitative conclusions (regarding H1–H4) hold across all 20 seeds

---

## File structure

```
abm-project/
├── agent.py           # Agent class: prototypes, labeling, update
├── environment.py     # Network construction, neighbor scheduling, stimulus sampling
├── metrics.py         # Pairwise consistency, overall consistency
├── simulation.py      # Main loop; accepts condition, update_rule, seed as arguments
├── run_simulation.py  # Simulation for all conditions * update strategies * seeds
├── tests/verify.py    # All verification checks
├── PROMPT.md
├── PLAN.md
├── SKILL.md
├── Dockerfile
└── README.md
```

---

