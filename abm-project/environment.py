"""Network construction, neighbor scheduling, and stimulus sampling."""

import math

import networkx as nx
import numpy as np

BA_M_VALUES = (1, 2, 5, 10)
NETWORK_CONDITIONS = ("lattice", "m1", "m2", "m5", "m10", "star")


def _lattice_dimensions(n):
    """Choose near-square rows/cols for a lattice with n nodes."""
    rows = int(math.floor(math.sqrt(n)))
    while rows > 1 and n % rows != 0:
        rows -= 1
    if rows < 1:
        rows = 1
    cols = n // rows
    if rows * cols != n:
        return None
    return rows, cols


def make_network(condition, n, rng):
    """
    Build a communication network for the given condition.

    condition: 'lattice', 'm1'..'m10', or 'star'
    """
    if condition == "star":
        return nx.star_graph(n - 1)

    if condition == "lattice":
        dims = _lattice_dimensions(n)
        if dims is None:
            graph = nx.path_graph(n)
        else:
            rows, cols = dims
            graph = nx.grid_2d_graph(rows, cols)
            graph = nx.convert_node_labels_to_integers(graph)
        return graph

    if condition.startswith("m") and condition[1:].isdigit():
        m = int(condition[1:])
        if m >= n:
            raise ValueError(f"BA parameter m={m} must be less than n={n}")
        seed = int(rng.integers(0, 2**31 - 1))
        return nx.barabasi_albert_graph(n, m, seed=seed)

    raise ValueError(f"Unknown network condition: {condition}")


def degree_centralization(graph):
    """
    Freeman degree centralization:
    C = sum(d_max - d_i) / ((n - 1)(n - 2))
    """
    n = graph.number_of_nodes()
    if n <= 2:
        return 0.0

    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
    d_max = degrees.max()
    numerator = np.sum(d_max - degrees)
    denominator = (n - 1) * (n - 2)
    return float(numerator / denominator)


def agent_degrees(graph):
    """Return node degrees as a list indexed by node id."""
    n = graph.number_of_nodes()
    degrees = [0] * n
    for node, degree in graph.degree():
        degrees[int(node)] = int(degree)
    return degrees


def hub_node(graph):
    """Return the node with maximum degree (lowest index on ties)."""
    degrees = dict(graph.degree())
    max_degree = max(degrees.values())
    for node in sorted(graph.nodes()):
        if degrees[node] == max_degree:
            return node
    return 0


def sample_pair(graph, rng):
    """Sample a directed (speaker, listener) pair from a random edge."""
    edges = list(graph.edges())
    if not edges:
        raise ValueError("Graph has no edges; cannot sample a communication pair.")

    speaker, listener = edges[int(rng.integers(0, len(edges)))]
    if rng.random() < 0.5:
        speaker, listener = listener, speaker
    return speaker, listener


def sample_stimulus(rng):
    """Sample a stimulus uniformly from [0, 1]^2."""
    return rng.uniform(0.0, 1.0, size=2)
