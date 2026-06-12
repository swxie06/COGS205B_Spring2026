"""Consistency metrics and prototype-based measurements."""

import numpy as np

TEST_STIMULI_SEED = 424242
N_TEST_STIMULI = 50
N_PAIRS = 100


def fixed_test_stimuli():
    """Return 50 fixed test stimuli shared across all simulation runs."""
    rng = np.random.default_rng(TEST_STIMULI_SEED)
    return rng.uniform(0.0, 1.0, size=(N_TEST_STIMULI, 2))


def pairwise_consistency(agents, test_stimuli, rng, n_pairs=N_PAIRS):
    """
    For each test stimulus, draw random agent pairs and measure label agreement.
    """
    n_agents = len(agents)
    if n_agents < 2:
        return 1.0

    per_stimulus = []
    for stimulus in test_stimuli:
        pair_agreements = []
        for _ in range(n_pairs):
            i, j = rng.choice(n_agents, size=2, replace=False)
            pair_agreements.append(agents[i].label(stimulus) == agents[j].label(stimulus))
        per_stimulus.append(float(np.mean(pair_agreements)))
    return float(np.mean(per_stimulus))


def overall_consistency(agents, test_stimuli):
    """
    For each stimulus, find the majority label and measure agreement with it.
    """
    n_agents = len(agents)
    if n_agents == 0:
        return 0.0

    per_stimulus = []
    for stimulus in test_stimuli:
        labels = np.array([agent.label(stimulus) for agent in agents])
        counts = np.bincount(labels, minlength=2)
        majority_label = int(np.argmax(counts))
        agreement = float(np.mean(labels == majority_label))
        per_stimulus.append(agreement)
    return float(np.mean(per_stimulus))


def population_prototypes(agents):
    """Average prototype vectors across agents for each category."""
    if not agents:
        return np.zeros((2, 2))
    stacked = np.stack([agent.prototypes for agent in agents], axis=0)
    return np.mean(stacked, axis=0)


def prototype_diversity(agents):
    """
    Standard deviation of prototype coordinates per category.

    Returns dict with per-category std (shape 2x2) and overall scalar mean.
    """
    if not agents:
        return {"per_category": np.zeros((2, 2)), "overall": 0.0}

    stacked = np.stack([agent.prototypes for agent in agents], axis=0)
    per_category = np.std(stacked, axis=0)
    overall = float(np.mean(per_category))
    return {"per_category": per_category, "overall": overall}


def prototype_set_distance(proto_a, proto_b):
    """Euclidean distance between two flattened prototype sets."""
    a = np.asarray(proto_a, dtype=float).reshape(-1)
    b = np.asarray(proto_b, dtype=float).reshape(-1)
    return float(np.linalg.norm(a - b))


def pairwise_prototype_distance_matrix(agents):
    """NxN matrix of pairwise prototype-set distances between agents."""
    n = len(agents)
    matrix = np.zeros((n, n), dtype=float)
    vectors = [agent.prototype_vector() for agent in agents]
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(vectors[i] - vectors[j]))
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix
