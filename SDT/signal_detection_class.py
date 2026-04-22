# Week 3 homework

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        if any(x < 0 for x in [hits, misses, false_alarms, correct_rejections]):
            raise ValueError("Input values must be 0 or greater.")
        if not all(isinstance(x, int) for x in [hits, misses, false_alarms, correct_rejections]):
            raise TypeError("Input values must be integers.")
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    # Core class
    def hit_rate(self):
        if self.hits + self.misses == 0:
            raise ValueError("Cannot divide by 0.")
        return self.hits / (self.hits + self.misses)
    def false_alarm_rate(self):
        if self.false_alarms + self.correct_rejections == 0:
            raise ValueError("Cannot divide by 0.")
        return self.false_alarms / (self.false_alarms + self.correct_rejections)
    def d_prime(self):
        H = self.hit_rate()
        FA = self.false_alarm_rate()
        if H == 0 or H == 1:
            raise ValueError(f"Hit rate is exactly {H}. Need to apply correction before computing d_prime.")
        if FA == 0 or FA == 1:
            raise ValueError(f"False alarm rate is exactly {FA}. Need to apply correction before computing d_prime.")
        return norm.ppf(H) - norm.ppf(FA)
    def criterion(self):
        H = self.hit_rate()
        FA = self.false_alarm_rate()
        if H == 0 or H == 1:
            raise ValueError(f"Hit rate is exactly {H}. Need to apply correction before computing criterion.")
        if FA == 0 or FA == 1:
            raise ValueError(f"False alarm rate is exactly {FA}. Need to apply correction before computing criterion.")
        return -0.5 * (norm.ppf(H) + norm.ppf(FA))
    
    # Operators
    def __add__(self, other):
        if not isinstance(other, SignalDetection):
            raise TypeError("Input arguments must be SignalDetection instances.")
        return SignalDetection(
            self.hits + other.hits, 
            self.misses + other.misses, 
            self.false_alarms + other.false_alarms, 
            self.correct_rejections + other.correct_rejections
        )
    def __sub__(self, other):
        # Left out as deliberate failure
        if not isinstance(other, SignalDetection):
            raise TypeError("Input arguments must be SignalDetection instances.")
        return SignalDetection(
            self.hits - other.hits, 
            self.misses - other.misses, 
            self.false_alarms - other.false_alarms, 
            self.correct_rejections - other.correct_rejections
        )
    def __mul__(self, factor):
        if not isinstance(factor, int):
            raise TypeError("Input factor must be an integer.")
        return SignalDetection(
            self.hits * factor, 
            self.misses * factor, 
            self.false_alarms * factor, 
            self.correct_rejections * factor
        )
    
    # Plotting
    def plot_sdt(self):
        d = self.d_prime()
        c = self.criterion()

        x = np.linspace(min(-4, d-4), max(4, d+4), 1000)

        noise = norm.pdf(x, 0, 1)
        signal = norm.pdf(x, d, 1)

        fig, ax = plt.subplots()

        ax.plot(x, noise, color="tab:blue", label="Noise")
        ax.plot(x, signal, color="tab:orange", label="Signal")

        ax.axvline(x=0, linestyle=":", linewidth=1.5, color="tab:blue")
        ax.axvline(x=d, linestyle=":", linewidth=1.5, color="tab:orange")

        ax.axvline(x=c, linestyle="--", color="tab:green", label="Criterion")

        ax.annotate('', xy=(d, 0.1), xytext=(0, 0.1),
                     arrowprops=dict(arrowstyle='<->'))
        ax.text(d/2, 0.12, "d'")

        ax.set_xlabel("Decision axis")
        ax.set_ylabel("Probability density")
        ax.set_title("SDT plot")
        ax.legend()
        
        return fig, ax
        

    @staticmethod
    def plot_roc(sdt_list):
        points = [(s.hit_rate(), s.false_alarm_rate()) for s in sdt_list]
        points = [(0, 0)] + points + [(1, 1)]
        
        # sort according to hit rate
        points = sorted(points, key=lambda x: x[0])
        hit_rates, fa_rates = zip(*points)

        fig, ax = plt.subplots()
        ax.plot(hit_rates, fa_rates, marker='o')

        ax.set_xlabel("Hit Rate")
        ax.set_ylabel("False Alarm Rate")
        ax.set_title("ROC Curve")
        
        return fig, ax