"""Agent model: prototypes, labeling, and prototype updates."""

import numpy as np

LABELS = ("Dax", "Leca")


class Agent:
    """An agent with two prototype vectors for binary labeling."""

    def __init__(self, agent_id, rng):
        self.agent_id = agent_id
        # Row 0 = Dax, row 1 = Leca; columns = (color, antenna)
        self.prototypes = rng.uniform(0.0, 1.0, size=(2, 2))

    def distances(self, x):
        """Euclidean distances from stimulus x to each prototype."""
        return np.linalg.norm(self.prototypes - x, axis=1)

    def label(self, x):
        """Assign label of nearest prototype; tie-break toward Dax (index 0)."""
        distances = self.distances(x)
        return int(np.argmin(distances))

    def margin(self, x):
        """Absolute difference in distance to the two prototypes."""
        distances = self.distances(x)
        return float(abs(distances[0] - distances[1]))

    def update(self, label_idx, x, eta):
        """Move the given label's prototype toward stimulus x."""
        self.prototypes[label_idx] = (
            (1.0 - eta) * self.prototypes[label_idx] + eta * x
        )

    def prototypes_in_range(self):
        """Return True if all prototype coordinates lie in [0, 1]."""
        return bool(np.all(self.prototypes >= 0.0) and np.all(self.prototypes <= 1.0))

    def copy_prototypes(self):
        """Return a copy of prototype vectors."""
        return self.prototypes.copy()

    def prototype_vector(self):
        """Flatten prototype vectors to a (4,) array for distance computations."""
        return self.prototypes.reshape(-1)


def create_agents(n, rng):
    """Create n agents with independently initialized prototypes."""
    return [Agent(i, rng) for i in range(n)]
