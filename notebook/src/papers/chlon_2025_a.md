# Chlon 2025 - LLMs are Bayesian in Expectation

[LLMs are Bayesian, in Expectation, not in Realization](https://arxiv.org/abs/2507.11768)

This paper uses information theoretic lens to analyze the effect of positional encoding on permutation qualities of LLMs and propose an algorithm to derive the optimal chain of thought length.

## Summary

In context learning allows LLMs to adapt to new tasks using only a few examples at inference time. A theoretical framework for interpreting ICL is through the lens of bayesian inference (see [Xie 2022 - ICL as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080)). It proposes that transformers implicitly perform posterior updates over a latent concept variable, with the pretraining distribution encoding a prior over possible tasks.

This perspective has been challenged by [Falck 2024 - Is ICL in LLMs bayesian?](https://arxiv.org/abs/2406.00793), which demonstrates empirically that transformer based language models systematically violate the martingale property. Specifically, for exchangeable data where the order of observations carries no information, bayesian posterior predictive distributions must satisfy:
$$
    \E[f(X_{n+1}) | X_1, ..., X_n] = \E[f(X_{n+1}) | X_{\pi(1)}, ..., X_{\pi(n)}]
$$

For any permutation $\pi$ and bounded function $f$. The experiments show that LLMs like GPT-3.5 consistently violate this property under input permutation.

This paper observes that while bayesian inference assumes exchangeable data, position encodings fundamentally break the symmetry. This is formalized using two complexity measures:
- Kolmogorov complexity $K(X)$ of a sequence, which is permutation invariant for exchangeable data
- Conditional complexity $K(X | \pi)$ given a specific ordering $\pi$

It then shows that transformers with position encoding minimizes:
$$
    \E_{\pi \sim \mathcal{U}(S_n)} [
        K(X|\pi)
    ] = K(X) + I(X; \pi)
$$

Where:
- $\mathcal{U}(S_n)$ denotes the uniform distribution over permutations consistent with sufficient statistics of the data
    - For iid data, we can just sample uniformly over all permutations
- $I(X;\pi)$ is the mutual information between sequences and their orderings

Note that this is a well known theorem (the kolomogorov version of Shannon's information identity) applied to this context.

## Notations

- $X = (x_1, ..., x_n)$ denotes a sequence of observations
- $S_n = \sum_{i=1}^n x_i$ is the sufficient statistic for bernoulli sequences
- $\pi \in S_n$ represents a permutation of $n$ elements
- $\mathcal{T}_\theta$ denotes a transformer with parameters $\theta$
- $K(X)$ is the kolmogorv complexity of sequence $X$
- $H(p) = -p \log p - (1-p) \log (1-p)$ is the binary entropy function
- $k$ denotes the number of chain of thought tokens
- $\epsilon$ denotes the target error tolerance 


