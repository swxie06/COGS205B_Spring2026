"""Main simulation loop for emergent label convergence."""

import numpy as np

from agent import create_agents
from environment import (
    agent_degrees,
    degree_centralization,
    hub_node,
    make_network,
    sample_pair,
    sample_stimulus,
)
from metrics import (
    fixed_test_stimuli,
    overall_consistency,
    pairwise_consistency,
    prototype_diversity,
)

UPDATE_RULES = ("unconditional", "conflict", "ambiguous")
DEFAULT_MEASURE_EVERY = 20
CONVERGENCE_THRESHOLD = 0.7


def should_update(listener, speaker_label, stimulus, update_rule, delta):
    """Decide whether the listener updates its prototypes for this event."""
    if update_rule == "unconditional":
        return True
    if update_rule == "conflict":
        listener_label = listener.label(stimulus)
        return speaker_label != listener_label
    if update_rule == "ambiguous":
        return listener.margin(stimulus) < delta
    raise ValueError(f"Unknown update_rule: {update_rule}")


def run_communication_event(agents, graph, rng, update_rule, eta, delta):
    """Execute one speaker-listener communication event."""
    speaker_id, listener_id = sample_pair(graph, rng)
    stimulus = sample_stimulus(rng)
    speaker = agents[speaker_id]
    listener = agents[listener_id]
    speaker_label = speaker.label(stimulus)

    if should_update(listener, speaker_label, stimulus, update_rule, delta):
        listener.update(speaker_label, stimulus, eta)


def prototypes_in_range(agents):
    """Check that all agents keep prototypes within [0, 1]."""
    return all(agent.prototypes_in_range() for agent in agents)


def measure(agents, test_stimuli, rng):
    """Compute consistency and prototype diversity metrics at the current timestep."""
    diversity = prototype_diversity(agents)
    return {
        "pairwise_consistency": pairwise_consistency(agents, test_stimuli, rng),
        "overall_consistency": overall_consistency(agents, test_stimuli),
        "prototype_diversity": diversity["overall"],
    }


def timesteps_to_threshold(trajectory, key, threshold=CONVERGENCE_THRESHOLD):
    """Return first timestep reaching threshold, or None if never reached."""
    for record in trajectory:
        if record[key] >= threshold:
            return record["timestep"]
    return None


def run(
    condition,
    update_rule,
    seed,
    n=100,
    T=1000,
    eta=0.1,
    delta=0.1,
    measure_every=DEFAULT_MEASURE_EVERY,
    track_prototypes=False,
):
    """
    Run one simulation for a network condition, update rule, and seed.

    Returns a dictionary with trajectories and end-of-run artifacts.
    """
    if update_rule not in UPDATE_RULES:
        raise ValueError(f"Unknown update_rule: {update_rule}")

    rng = np.random.default_rng(seed)
    metric_rng = np.random.default_rng(seed + 1)
    graph = make_network(condition, n, rng)
    agents = create_agents(n, rng)
    test_stimuli = fixed_test_stimuli()

    centralization = degree_centralization(graph)
    hub_id = hub_node(graph)
    degrees = agent_degrees(graph)
    initial_prototypes = [agent.copy_prototypes().tolist() for agent in agents]
    hub_initial_prototypes = agents[hub_id].copy_prototypes()

    trajectory = []
    prototype_snapshots = [] if track_prototypes else None

    for timestep in range(T + 1):
        if timestep % measure_every == 0:
            metrics = measure(agents, test_stimuli, metric_rng)
            trajectory.append(
                {
                    "timestep": timestep,
                    "pairwise_consistency": metrics["pairwise_consistency"],
                    "overall_consistency": metrics["overall_consistency"],
                    "prototype_diversity": metrics["prototype_diversity"],
                }
            )
            if track_prototypes:
                prototype_snapshots.append(
                    [agent.copy_prototypes() for agent in agents]
                )

        if timestep == T:
            break

        for _ in range(n):
            run_communication_event(agents, graph, rng, update_rule, eta, delta)
            if not prototypes_in_range(agents):
                raise RuntimeError(
                    f"Prototype out of [0,1] at timestep {timestep} for seed {seed}"
                )

    final_prototypes = [agent.copy_prototypes().tolist() for agent in agents]
    final_diversity = prototype_diversity(agents)
    final_metrics = trajectory[-1]

    return {
        "condition": condition,
        "update_rule": update_rule,
        "seed": seed,
        "n": n,
        "T": T,
        "eta": eta,
        "delta": delta,
        "centralization": centralization,
        "hub_id": hub_id,
        "agent_degrees": degrees,
        "hub_initial_prototypes": hub_initial_prototypes.tolist(),
        "hub_final_prototypes": agents[hub_id].copy_prototypes().tolist(),
        "initial_prototypes": initial_prototypes,
        "final_prototypes": final_prototypes,
        "trajectory": trajectory,
        "final_pairwise_consistency": final_metrics["pairwise_consistency"],
        "final_overall_consistency": final_metrics["overall_consistency"],
        "final_prototype_diversity": final_diversity["overall"],
        "convergence_timestep_pairwise": timesteps_to_threshold(
            trajectory, "pairwise_consistency"
        ),
        "convergence_timestep_overall": timesteps_to_threshold(
            trajectory, "overall_consistency"
        ),
        "prototype_snapshots": prototype_snapshots,
    }


def run_deterministic_copy(condition, update_rule, seed, **kwargs):
    """Convenience helper for tests comparing repeated runs."""
    first = run(condition, update_rule, seed, **kwargs)
    second = run(condition, update_rule, seed, **kwargs)
    return first, second
