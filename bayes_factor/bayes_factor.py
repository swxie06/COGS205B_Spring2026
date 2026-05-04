# Week 4 homework
import math
import scipy

class BayesFactor:
    def __init__(self, n, k):
        if (not isinstance(n, int)) or (not isinstance(k, int)):
            raise TypeError("Input values must be integers.")
        if n < 0 or k < 0:
            raise ValueError("Input values must be non-negative.")
        if k > n:
            raise ValueError("Input k cannot be greater than n.")
        
        self.n = n
        self.k = k
        self.a = 0.4999
        self.b = 0.5001

    def likelihood(self, theta): 
        # Left as intentionally failing test
        if theta < 0:
            raise ValueError("Theta must be between 0 and 1.") 
        # Correct version below
        # if theta < 0 or theta > 1:
        #     raise ValueError("Theta must be between 0 and 1.")

        n, k = self.n, self.k
        p = math.comb(n, k) * (theta ** k) * ((1 - theta) ** (n - k))
        return p
    
    def integrand_slab(self, theta):
        return self.likelihood(theta)
    def evidence_slab(self): 
        res, _ = scipy.integrate.quad(self.integrand_slab, 0, 1)
        return res

    def integrand_spike(self, theta):
        width = self.b - self.a
        return self.likelihood(theta) / width
    def evidence_spike(self):
        res, _ = scipy.integrate.quad(self.integrand_spike, self.a, self.b)
        return res
    
    def bayes_factor(self):
        if self.evidence_slab() == 0:
            raise ValueError("Cannot divide by 0.")
        return self.evidence_spike() / self.evidence_slab()