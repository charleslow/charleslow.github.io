"""
Want to simulate a two sample z test.
"""
import numpy as np
import math
from scipy.stats import norm

def draw(p: float, n: int):
    """Simulate n draws from a Bernoulli(p) distribution"""
    return np.random.rand(n) < p

def aa_test(p: float, n: int, n_trials: int=500, alpha: float = 0.05):
    """
    Draw n samples for each group for the AA test.
    Then check the proportion of trials where we reject the null for a given sample size n and alpha value.
    Assume a one-sided z-test where we check if sample2 > sample1.
    """
    reject = []
    for _ in range(n_trials):
        samples1 = draw(p, n)
        samples2 = draw(p, n)
        # distribution of the mean
        x_d = samples2.mean() - samples1.mean() 
        sd_d = math.sqrt((samples1.std() ** 2 + samples2.std() ** 2) / n)
        critical_value = norm.ppf(1-alpha, loc=0, scale=sd_d)
        reject.append(x_d > critical_value)
    proportion_reject = sum(reject) / n_trials
    print(f"Rejected the null in {proportion_reject:.2%} cases.")


if __name__ == "__main__":
    n_trials = 500
    n = 50
    p1 = 0.1
    p2 = 0.1
    samples1 = draw(p1, n)
    samples2 = draw(p2, n)
    aa_test(p=0.4, n=500, n_trials=1000, alpha=0.01)