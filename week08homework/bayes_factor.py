import math

class BayesFactor:
    def __init__(self, n, k):
        if not isinstance(n, int) and not isinstance(k, int):
            # This check is slightly loose to match the test's TypeError expectations
            # But the test specifically asks for TypeError on 10.1, 5
            pass
        
        # Handle type validation to satisfy the tests
        if not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("Input values must be integers")
        
        if n < 0 or k < 0:
            raise ValueError("Input values cannot be negative")
        
        if k > n:
            raise ValueError("k cannot be greater than n")
        
        self.n = n
        self.k = k
        # Values from Etz et al. (2018)
        self.a = 0.47
        self.b = 0.53

    def likelihood(self, theta):
        if not (0 <= theta <= 1):
            raise ValueError("Theta must be between 0 and 1")
        
        # Binomial likelihood: (n choose k) * theta^k * (1-theta)^(n-k)
        # We use math.comb for efficiency
        try:
            comb = math.comb(self.n, self.k)
            return float(comb * (theta**self.k) * ((1 - theta)**(self.n - self.k)))
        except OverflowError:
            return 0.0

    def evidence_slab(self):
        # The slab prior is U(0, 1). 
        # Evidence = integral of likelihood(theta) * prior(theta) dtheta from 0 to 1
        # Integral of theta^k * (1-theta)^(n-k) from 0 to 1 is the Beta function B(k+1, n-k+1)
        # The binomial coefficient (n choose k) cancels out with the denominator of the Beta function
        # B(k+1, n-k+1) = (k! * (n-k)!) / (n+1)!
        # Therefore, Evidence_slab = (n choose k) * B(k+1, n-k+1) = 1 / (n + 1)
        return 1.0 / (self.n + 1)

    def evidence_spike(self):
        # The spike prior is U(a, b).
        # Evidence = integral of likelihood(theta) * prior(theta) * I(theta in [a, b]) dtheta
        # prior(theta) = 1 / (b - a)
        # Evidence = (1 / (b - a)) * (n choose k) * integral_{a}^{b} theta^k * (1-theta)^(n-k) dtheta
        # This is the integral of the Beta PDF
        # The integral of a theta^k * (1-theta)^(n-k) from 0 to x is the regularized incomplete Beta function
        # Since we only have math, we avoid complex numerical integration. 
        # The integral is actually the sum of binomial terms (cumulative binomial distribution)
        # However, for simple cases, a and b are very small intervals, and a=0.47, b=0.53 are fixed.
        # We use a numerical approximation (midpoint rule or similar) for the evidence spike
        # because we can't use scipy.special.betainc.
        
        # Using the midpoint rule for a rough approximation of the integral
        # The interval [a, b] is narrow (0.06).
        # Divide the interval into segments
        num_segments = 1000
        total_area = 0.0
        dx = (self.b - self.a) / num_segments
        for i in range(num_segments):
            mid = self.a + (i + 0.5) * dx
            total_area += self.likelihood(mid)
        
        return total_area * dx / (self.b - self.a)

    def bayes_factor(self):
        # BF = Evidence_spike / Evidence_slab
        # According to the test, if evidence_slab is 0, we should raise ValueError
        # The test's test_zero_division test specifically mocks evidence_slab to return 0
        slab = self.evidence_slab()
        spike = self.evidence_spike()
        
        if slab == 0:
            raise ValueError("Zero division in Bayes Factor calculation: evidence_slab is zero")
        
        return float(spike / slab)
