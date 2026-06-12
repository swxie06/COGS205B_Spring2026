## Model Specification

This project investigates the emergence of shared category labels through repeated communication in a population of agents. Agents interact over a network and attempt to coordinate on labels for stimuli without access to any ground-truth category structure.

### Agents and State Variables

The environment is a two-dimensional feature space consisting of two continuous features bounded between 0 and 1. Each agent maintains two category prototypes in the feature space, corresponding to two labels (e.g., _Dax_ and _Leca_). Agents classify a stimulus by assigning the label whose prototype is closest in Euclidean distance.

### Communication and Update Rules

During each communication event, a speaker generates a random stimulus, labels it according to its current prototypes, and sends the stimulus-label pair to a neighboring listener. The listener may update its prototype associated with the communicated label using a simple prototype-learning rule:
$$\text{prototype}_{label} \leftarrow (1-\eta) \text{ prototype}_{label} + \eta X$$
where $X$ is the stimulus, and $\eta$ is the learning rate.

Three listener update strategies are compared:

- **Unconditional:** always update.
- **Conflict-driven:** update only when the listener disagrees with the speaker's label.
- **Ambiguity-driven:** update only when the listener's classification is sufficiently uncertain.

Agents interact on networks with varying levels of degree centralization, ranging from lattice-like structures to highly centralized star networks.

### Metrics

To quantify convergence and coordination, the model records:

- **Pairwise Consistency:** the probability that two randomly selected agents assign the same label to a stimulus.
- **Overall Consistency:** the proportion of agent classifications that agree with the population majority label for each stimulus.
- **Prototype Diversity:** variation in prototype locations across agents.

These metrics are tracked throughout the simulation to evaluate how network centralization and update strategies influence the formation of the shared emergent labeling system.

## Results

The simulations show that a shared labeling system can emerge through communication alone, but the speed and extent of convergence depend strongly on both network structure and listener update strategy.

### Hypothesis 1: Centralization and Convergence

Overall, the results provide partial support for the hypothesis that greater network centralization accelerates convergence. Under the **unconditional** update rule, highly centralized networks reached high levels of pairwise and overall consistency substantially faster than lattice-like networks. Similar trends were observed for the **ambiguous** update rule. However, the relationship was not strictly monotonic, and under the **conflict-driven** update rule many conditions failed to reach the convergence threshold within the simulation.

These findings suggest that network centralization facilitates convergence, but its effect depends on the listener update strategy.

### Hypothesis 2: Hub Bias

The analysis did not reveal a strong relationship between agent degree and similarity between an agent's initial prototypes and the final population prototypes. While highly connected agents may exert greater influence on the communication process, the results do not provide clear evidence that final conventions are biased toward their initial prototypes.

### Hypothesis 3: Dialects and Prototype Diversity

The results provide partial support for the hypothesis that low-centralization networks would sustain localized "dialects". Low-centralization networks showed higher prototype diversity and lower consistency than highly centralized networks. Pairwise prototype distance matrices also showed greater heterogeneity in lattice networks than in star networks, indicating that different parts of the population maintained more distinct prototype systems.

### Hypothesis 4: Update Strategies

Update strategy had a large effect on convergence. The **unconditional** rule consistently produced the fastest convergence and the highest final consistency. The **conflict-driven** rule was the most conservative and often failed to achieve substantial convergence. The **ambiguity-driven** rule produced intermediate outcomes, reaching moderate levels of consistency while preserving more prototype diversity than unconditional updating.

## Reflection

How I ensured accuracy of the codebase:

- The agent was required to implement several verification checks before running the simulation. These included invariant tests, deterministic seed tests, ablation tests, edge case tests, and sensitivity tests. I manually reviewed the verification code to ensure that the tests were implemented correctly and that the agent was not cheating.
- To reduce the influence of stochastic variation and support reproducibility, all reported results were aggregated across 20 random seeds rather than relying on individual simulation runs. All sources of randomness were controlled through explicit seed management.
- A project-specific `SKILL.md` file was used to maintain consistent assumptions about the experimental design, hypotheses, verification requirements, and reporting standards throughout the AI-assisted development process.

Overall, I trust both the implementation and the main qualitative findings. The results were replicated across multiple seeds and were generally consistent with the original hypotheses. However, I have less confidence in the hub-bias hypothesis, which received little support in the current simulations. Additional experiments and analyses would be needed to determine whether the effect is genuinely absent or whether the current measurements failed to capture it adequately.