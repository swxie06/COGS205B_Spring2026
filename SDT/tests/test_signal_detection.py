import unittest
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from signal_detection_class import SignalDetection 

# Part 1: core SDT math tests
class TestSDTMath(unittest.TestCase):
    def setUp(self):
        self.sdt = SignalDetection(hits=80, misses=20, 
                                   false_alarms=20, correct_rejections=80)

    def test_hit_rate(self):
        self.assertEqual(self.sdt.hit_rate(), 0.8)
    def test_false_alarm_rate(self):
        self.assertEqual(self.sdt.false_alarm_rate(), 0.2)
    def test_d_prime(self):
        expected = norm.ppf(0.8) - norm.ppf(0.2)
        self.assertAlmostEqual(self.sdt.d_prime(), expected)
    def test_criterion(self):
        expected = -0.5 * (norm.ppf(0.8) + norm.ppf(0.2))
        self.assertAlmostEqual(self.sdt.criterion(), expected)
    
    def test_hit_rate_divided_by_zero(self):
        sdt = SignalDetection(hits=0, misses=0, 
                              false_alarms=2, correct_rejections=8)
        with self.assertRaises(ValueError):
            sdt.hit_rate()
    def test_false_alarm_rate_divided_by_zero(self):
        sdt = SignalDetection(hits=8, misses=2, 
                              false_alarms=0, correct_rejections=0)
        with self.assertRaises(ValueError):
            sdt.false_alarm_rate()
    
    def test_d_prime_extreme_values_h(self):
        sdt = SignalDetection(hits=10, misses=0, 
                              false_alarms=2, correct_rejections=8)
        with self.assertRaises(ValueError):
            sdt.d_prime()
    def test_d_prime_extreme_values_fa(self):
        sdt = SignalDetection(hits=8, misses=2, 
                              false_alarms=0, correct_rejections=10)
        with self.assertRaises(ValueError):
            sdt.d_prime()
    def test_criterion_extreme_values_h(self):
        sdt = SignalDetection(hits=0, misses=10, 
                              false_alarms=2, correct_rejections=8)
        with self.assertRaises(ValueError):
            sdt.criterion()
    def test_criterion_extreme_values_fa(self):
        sdt = SignalDetection(hits=8, misses=2, 
                              false_alarms=10, correct_rejections=0)
        with self.assertRaises(ValueError):
            sdt.criterion()


# Part 2: input validation and object safety
class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.sdt = SignalDetection(hits=10, misses=5, 
                              false_alarms=3, correct_rejections=7)

    def test_input_negative_counts(self):
        with self.assertRaises(ValueError):
            SignalDetection(hits=-1, misses=5,
                            false_alarms=5, correct_rejections=5)
            
    def test_input_wrong_types(self):
        with self.assertRaises(TypeError):
            SignalDetection(hits=5, misses=5,
                            false_alarms=5, correct_rejections=1.5)
    
    def test_add_argument_type(self):
        with self.assertRaises(TypeError):
            sdt_res = self.sdt + 5
    def test_sub_argument_type(self):
        with self.assertRaises(TypeError):
            sdt_res = self.sdt - 5
    def test_mul_argument_type(self):
        with self.assertRaises(TypeError):
            sdt_res = self.sdt * self.sdt



# Part 3: operator behavior
class TestOperatorBehavior(unittest.TestCase):
    def setUp(self):
        self.sdt1 = SignalDetection(hits=10, misses=5,
                                 false_alarms=3, correct_rejections=7)
        self.sdt2 = SignalDetection(hits=2, misses=1,
                                 false_alarms=1, correct_rejections=2)
    
    def test_add(self):
        sdt_res = self.sdt1 + self.sdt2
        self.assertEqual(sdt_res.hits, 10+2)
        self.assertEqual(sdt_res.misses, 5+1)
        self.assertEqual(sdt_res.false_alarms, 3+1)
        self.assertEqual(sdt_res.correct_rejections, 7+2)
    def test_add_non_mutation(self):
        sdt_res = self.sdt1 + self.sdt2
        self.assertEqual(self.sdt1.hits, 10)
        self.assertEqual(self.sdt2.hits, 2)
    def test_add_result_consistency(self):
        sdt_res = self.sdt1 + self.sdt2
        sdt_expected = SignalDetection(hits=10+2, misses=5+1,
                                 false_alarms=3+1, correct_rejections=7+2)
        self.assertEqual(sdt_res.hit_rate(), sdt_expected.hit_rate())


    def test_sub(self):
        sdt_res = self.sdt1 - self.sdt2
        self.assertEqual(sdt_res.hits, 10-2)
        self.assertEqual(sdt_res.misses, 5-1)
        self.assertEqual(sdt_res.false_alarms, 3-1)
        self.assertEqual(sdt_res.correct_rejections, 7-2)
    def test_sub_non_mutation(self):
        sdt_res = self.sdt1 - self.sdt2
        self.assertEqual(self.sdt1.hits, 10)
        self.assertEqual(self.sdt2.hits, 2)
    def test_sub_result_consistency(self):
        sdt_res = self.sdt1 - self.sdt2
        sdt_expected = SignalDetection(hits=10-2, misses=5-1,
                                 false_alarms=3-1, correct_rejections=7-2)
        self.assertEqual(sdt_res.hit_rate(), sdt_expected.hit_rate())
    def test_sub_negative_results(self):
        a = SignalDetection(hits=5,  misses=5,  false_alarms=5,  correct_rejections=5)
        b = SignalDetection(hits=10, misses=10, false_alarms=10, correct_rejections=10)
        with self.assertRaises(ValueError):
            sdt_res = a - b

    def test_mul(self):
        sdt_res = self.sdt1 * 3
        self.assertEqual(sdt_res.hits, 10*3)
        self.assertEqual(sdt_res.misses, 5*3)
        self.assertEqual(sdt_res.false_alarms, 3*3)
        self.assertEqual(sdt_res.correct_rejections, 7*3)
    def test_mul_non_mutation(self):
        sdt_res = self.sdt1 * 3
        self.assertEqual(self.sdt1.hits, 10)
        self.assertEqual(self.sdt2.hits, 2)
    def test_sub_result_consistency(self):
        sdt_res = self.sdt1 * 3
        sdt_expected = SignalDetection(hits=10*3, misses=5*3,
                                 false_alarms=3*3, correct_rejections=7*3)
        self.assertEqual(sdt_res.hit_rate(), sdt_expected.hit_rate())

# Part 4: plotting behavior
class TestPlottingBehavior(unittest.TestCase):
    def setUp(self):
        self.sdt = SignalDetection(hits=80, misses=20,
                                   false_alarms=20, correct_rejections=80)
    
    def test_plot_sdt_returns_correct_objects(self):
        fig, ax = self.sdt.plot_sdt()
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
    def test_plot_sdt_has_labels(self):
        fig, ax = self.sdt.plot_sdt()
        self.assertIsNotNone(ax.get_xlabel())
        self.assertIsNotNone(ax.get_ylabel())
    def test_plot_sdt_has_legends(self):
        fig, ax = self.sdt.plot_sdt()
        self.assertIsNotNone(ax.get_legend())