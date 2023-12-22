# Sigmoid

## Sigmoid Relationship to Bradley-Terry Model

Suppose we have scores $s_i, s_j$ for objects $i$ and $j$. Under the Bradley-Terry model, we can express the preference probability as follows:

$$
\begin{aligned}
P(i \triangleright j) &= \frac{e^{s_i}}{e^{s_i} + e^{s_j}}\\
    &= \frac{1}{(e^{s_i} + e^{s_j}) / e^{s_i}}\\
    &= \frac{1}{1 + e^{s_j} / e^{s_i}}\\
    &= \frac{1}{1 + e^{s_j - s_i}}\\
    &= \frac{1}{1 + e^{-(s_i - s_j)}}\\
    &= \sigma(s_i - s_j)
\end{aligned}
$$

Where $\sigma(x) = 1 / (1 + e^{-x})$ is the sigmoid function.


