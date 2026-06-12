## Purpose

This project investigates whether communication pressure alone can produce an emergent shared label system among agents, and how network structure and update strategies influence convergence dynamics.

The primary goal is scientific validity and reproducibility, not software engineering sophistication.

---

## Core Research Principles

Always prioritize:

1. Correct implementation of the experimental design.
2. Reproducibility across random seeds.
3. Interpretability of results.
4. Simplicity and transparency of code.

Do not sacrifice experimental validity for computational efficiency or architectural elegance.

---

## Experimental Constraints

Treat the following components as part of the experimental design and do not modify them unless explicitly instructed:

- Hypotheses H1–H4
- Stimulus space definition
- Communication protocol
- Update rule definitions
- Network centralization manipulation
- Consistency metric definitions
- Verification requirements

If a proposed change would alter the scientific meaning of the experiment, explain the consequences before implementing it.

---

## Reproducibility Requirements

All stochastic processes must be controlled by explicit random seeds.

When adding new sources of randomness:

- expose them through configuration
- ensure deterministic behavior under a fixed seed
- verify that repeated runs produce identical outputs

Never introduce hidden randomness.

---

## Coding Guidelines

Prefer:

- straightforward implementations
- readable code
- explicit variable names
- modular functions with clear responsibilities

Avoid:

- unnecessary abstraction
- premature optimization
- complex inheritance hierarchies
- framework-heavy solutions

This is a research codebase rather than a production software system.

---

## Scientific Analysis Standards

Never draw conclusions from a single simulation run. All reported findings should be based on aggregation across multiple seeds.

When summarizing results:

- report means and variability
- identify possible stochastic effects
- distinguish robust patterns from noise

---

## Reporting Results

When generating summaries, reports, or figures:

1. State the experimental condition.
2. State the update rule.
3. Report the number of seeds.
4. Report the relevant consistency metric.
5. Separate observations from interpretations.

Use language appropriate for cognitive science and computational modeling research.