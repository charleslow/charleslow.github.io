"""
We wish to replicate the calculator of https://www.evanmiller.org/ab-testing/sample-size.html.

https://towardsdatascience.com/probing-into-minimum-sample-size-formula-derivation-and-usage-8db9a556280b
https://www.moresteam.com/whitepapers/download/power-stat-test.pdf
"""

from scipy.stats import norm
import math


def compute_minimal_sample_size(
    alpha: float,
    power: float,
    baseline_conversion_rate: float,
    minimum_detectable_effect: float,
):
    """
    Compute the minimal sample size for an A/B test. This assumes that the metric of interest
    is a bernoulli random variable.

    Parameters
    ----------
    alpha : float
        The significance level of the test.
    power : float
        The power of the test.
    baseline_conversion_rate : float
        The conversion rate of the baseline.
    minimum_detectable_effect : float
        The minimum detectable effect.

    Returns
    -------
    int
        The minimal sample size.
    """
    p = baseline_conversion_rate
    delta = minimum_detectable_effect * p
    p2 = p + delta

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # Let X1 ~ Bernoulli(p)
    # Then sd1 is the standard deviation of Y ~ X1 - X1' (think of an AA test)
    sd1 = math.sqrt(2 * p * (1.0 - p))

    # Let X2 ~ Bernoulli(p + delta)
    # Then sd2: stddev of Y ~ X2 - X1 (AB test)
    sd2 = math.sqrt(p * (1.0 - p) + p2 * (1.0 - p2))
    n = (z_alpha * sd1 + z_beta * sd2) ** 2 / delta ** 2
    
    return int(n)

if __name__ == "__main__":
    alpha = 0.05
    power = 0.95
    baseline_conversion_rate = 0.1
    minimum_detectable_effect = 0.2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    print(z_alpha, z_beta)