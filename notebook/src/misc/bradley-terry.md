# Bradley-Terry Model

Based on [wikipedia](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model). The Bradley-Terry model (1952) is a probability model that allows us to infer scores for individual objects based on a dataset of pairwise comparisons between them. Specifically, it estimates the probability that $i \triangleright j$ (i.e. $i$ is preferred to $j$) as:

$$
P(i \triangleright j) = \frac{p_i}{p_i + p_j}
$$

Where $p_i$ is a positive real-valued score assigned to object $i$ (not necessarily a probability). Typically, $p_i$ is parametrized as an exponential score $p_i = e^{\beta_i}$, and the goal is to learn the parameters $\beta_i$ from pairwise comparisons. This results in:

$$
P(i \triangleright j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}
$$

## Parameter Estimation

Parameter estimation is typically done using maximum likelihood. Starting with a set of pairwise comparisons between individual objects, let $w_{ij}$ be the number of times object $i$ beats object $j$. Then the likelihood of a given set of parameters $\mathbf{p} := [p_1, ..., p_n]$ ($n$ denotes number of objects) is as follows:

$$
\begin{aligned}
    L(\mathbf{p}) &= ln \prod_{ij} P(i \triangleright j)^{w_{ij}}\\
    &= \sum_{i=1}^n \sum_{j=1}^n ln \left( \frac{p_i}{p_i + p_j} \right)^{w_{ij}}\\
    &= \sum_{i=1}^n \sum_{j=1}^n w_{ij} \left[ ln p_i - ln(p_i + p_j) \right]\\
\end{aligned}
$$

This likelihood function can then be minimized by differentiating wrt $p_i$ and solved by setting to zero.

## BT Model as a Sigmoid Function

We can also express the likelihood as a function of the difference in scores $\beta_i - \beta_j$. Recall that the sigmoid function is $\sigma(x) = 1 / (1 + e^{-x})$. Then:
$$
\begin{aligned}
    L(\mathbf{p})
    &= \sum_{i=1}^n \sum_{j=1}^n w_{ij} \cdot ln \left[\frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} \right]\\
    &=\sum_{i=1}^n \sum_{j=1}^n w_{ij} \cdot ln\ \sigma(\beta_i - \beta_j) 
\end{aligned}
$$

The derivation for the second line above is found at [/Identities/sigmoid](../identities/sigmoid.md). This re-parametrization shows that the BT-model is really modelling the preference as a difference in scores and then running that through the sigmoid function to convert it into a probability. This means that we are basically running a pairwise logistic regression.