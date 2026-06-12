"""Verification checks for the emergent label ABM."""

import unittest

import numpy as np

from environment import degree_centralization, make_network
from metrics import prototype_set_distance
from simulation import run


class TestPrototypeInvariant(unittest.TestCase):
    def test_prototypes_remain_in_unit_square(self):
        result = run(
            condition="m2",
            update_rule="unconditional",
            seed=7,
            n=20,
            T=200,
            track_prototypes=False,
        )
        tracked = run(
            condition="m2",
            update_rule="unconditional",
            seed=7,
            n=20,
            T=200,
            track_prototypes=True,
        )
        for snapshot in tracked["prototype_snapshots"]:
            for agent_prototypes in snapshot:
                self.assertTrue(np.all(agent_prototypes >= 0.0))
                self.assertTrue(np.all(agent_prototypes <= 1.0))
        self.assertGreater(result["final_pairwise_consistency"], 0.0)


class TestDeterministicSeed(unittest.TestCase):
    def test_same_seed_same_results(self):
        kwargs = dict(condition="m1", update_rule="conflict", seed=123, n=30, T=100)
        first = run(**kwargs)
        second = run(**kwargs)
        self.assertEqual(first["trajectory"], second["trajectory"])
        self.assertEqual(first["final_prototypes"], second["final_prototypes"])
        self.assertEqual(
            first["final_prototype_diversity"], second["final_prototype_diversity"]
        )


class TestNoLearningAblation(unittest.TestCase):
    def test_eta_zero_pairwise_near_chance(self):
        result = run(
            condition="m5",
            update_rule="unconditional",
            seed=99,
            n=40,
            T=200,
            eta=0.0,
        )
        for record in result["trajectory"]:
            self.assertAlmostEqual(record["pairwise_consistency"], 0.5, delta=0.15)


class TestAmbiguityAblation(unittest.TestCase):
    def test_delta_zero_prevents_updates(self):
        result = run(
            condition="lattice",
            update_rule="ambiguous",
            seed=5,
            n=20,
            T=100,
            delta=0.0,
            track_prototypes=True,
        )
        first_snapshot = result["prototype_snapshots"][0]
        last_snapshot = result["prototype_snapshots"][-1]
        for initial, final in zip(first_snapshot, last_snapshot):
            np.testing.assert_allclose(initial, final)
        self.assertEqual(result["convergence_timestep_pairwise"], None)


class TestTwoAgentConvergence(unittest.TestCase):
    def test_two_agents_converge_to_identical_prototypes(self):
        result = run(
            condition="star",
            update_rule="unconditional",
            seed=10,
            n=2,
            T=500,
            eta=0.2,
        )
        distance = prototype_set_distance(
            result["final_prototypes"][0],
            result["final_prototypes"][1],
        )
        self.assertLess(distance, 0.05)


class TestSensitivityAcrossSeeds(unittest.TestCase):
    def test_qualitative_hypotheses_hold_across_seeds(self):
        seeds = range(5)
        h1_wins = 0
        h4_unconditional_wins = 0

        for seed in seeds:
            low = run(condition="lattice", update_rule="unconditional", seed=seed, n=40, T=500)
            high = run(condition="star", update_rule="unconditional", seed=seed, n=40, T=500)
            low_t = low["convergence_timestep_pairwise"]
            high_t = high["convergence_timestep_pairwise"]
            if high_t is not None and (low_t is None or high_t < low_t):
                h1_wins += 1
            elif high["final_pairwise_consistency"] > low["final_pairwise_consistency"]:
                h1_wins += 1

            unconditional = run(
                condition="m2", update_rule="unconditional", seed=seed, n=40, T=500
            )
            ambiguous = run(
                condition="m2", update_rule="ambiguous", seed=seed, n=40, T=500
            )
            u_t = unconditional["convergence_timestep_pairwise"]
            a_t = ambiguous["convergence_timestep_pairwise"]
            if u_t is not None and a_t is not None and u_t < a_t:
                h4_unconditional_wins += 1
            elif u_t is not None and a_t is None:
                h4_unconditional_wins += 1

        self.assertGreaterEqual(h1_wins, 3)
        self.assertGreaterEqual(h4_unconditional_wins, 3)


class TestNetworkCentralization(unittest.TestCase):
    def test_star_has_unit_centralization(self):
        rng = np.random.default_rng(0)
        graph = make_network("star", 20, rng)
        self.assertAlmostEqual(degree_centralization(graph), 1.0, places=5)

    def test_lattice_has_low_centralization(self):
        rng = np.random.default_rng(0)
        graph = make_network("lattice", 100, rng)
        self.assertLess(degree_centralization(graph), 0.1)


if __name__ == "__main__":
    unittest.main()
