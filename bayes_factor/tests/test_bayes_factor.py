import unittest

from bayes_factor import BayesFactor

class TestInput(unittest.TestCase):
    def setUp(self):
        self.bf = BayesFactor(10, 5)
    
    # 1. Input and state validation
    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            BayesFactor(10.1, 5)
    def test_negative_values(self):
        with self.assertRaises(ValueError) as context:
            BayesFactor(-10, 5)
        # Verify exception message
        self.assertIn("Input values", str(context.exception))
    def test_impossible_binomial(self): # k > n
        with self.assertRaises(ValueError) as context:
            BayesFactor(5, 10)
        # Verify exception message
        self.assertIn("k", str(context.exception))
    def test_object_consistency(self):
        self.assertEqual(self.bf.n, 10)
        self.assertEqual(self.bf.k, 5)
        self.assertEqual(self.bf.a, 0.4999)
        self.assertEqual(self.bf.b, 0.5001)
    
    # 2. API behavior and return contracts
    def test_methods_exist(self):
        self.assertTrue(callable(self.bf.likelihood))
        self.assertTrue(callable(self.bf.evidence_slab))
        self.assertTrue(callable(self.bf.evidence_spike))
        self.assertTrue(callable(self.bf.bayes_factor))
    def test_method_return_types(self):
        self.assertIsInstance(self.bf.likelihood(0.5), float)
        self.assertIsInstance(self.bf.evidence_slab(), float)
        self.assertIsInstance(self.bf.evidence_spike(), float)
        self.assertIsInstance(self.bf.bayes_factor(), float)
    
    # 3. Mathematical consistency checks
    # TODO
    def test_likelihood_behavior_theta_0(self):
        self.assertEqual(self.bf.likelihood(0), 0)
    def test_likelihood_behavior_theta_1(self):
        self.assertEqual(self.bf.likelihood(1), 0)
    def test_likelihood_peak(self):
        # likelihood should peak around theta = k/n
        peak = self.bf.likelihood(5/10)
        off = self.bf.likelihood(0.1)
        self.assertGreater(peak, off)
    def test_bayes_factor_extreme_cases(self):
        bf1 = BayesFactor(1, 0)
        bf2 = BayesFactor(10, 10)
        self.assertGreaterEqual(bf1.bayes_factor(), 0)
        self.assertGreaterEqual(bf2.bayes_factor(), 0)
    def test_evidence_consistency(self): 
        # evidence should be >= 0
        slab = self.bf.evidence_slab()
        spike = self.bf.evidence_spike()
        self.assertGreaterEqual(slab, 0)
        self.assertGreaterEqual(spike, 0)

    # 4. Error behavior
    def test_invalid_theta_small(self):
        with self.assertRaises(ValueError) as context:
            self.bf.likelihood(-0.1)
        # Verify exception message
        self.assertIn("Theta", str(context.exception))
    def test_invalid_theta_large(self):
        with self.assertRaises(ValueError) as context:
            self.bf.likelihood(1.1)
        # Verify exception message
        self.assertIn("Theta", str(context.exception))
    def test_zero_division(self):
        bf = BayesFactor(0, 0)
        bf.evidence_slab = lambda: 0
        with self.assertRaises(ValueError):
            bf.bayes_factor()