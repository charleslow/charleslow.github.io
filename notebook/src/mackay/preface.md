# Preface

The book provides some math as preface to understand the first chapter.

## The Binomial Distribution

### Example 1.1 - Binomial

> A coin has probability $f$ of coming up heads. The coin is tossed $N$ times. What is the probability distribution of the number of heads, $r$? What are the mean and variance of $r$?

The number of heads has a binomial distribution. Each instance has probability as such, and there are $\binom{N}{r}$ ways that it can occur.
$$
    P(r | f, N) = \binom{N}{r} f^r (1-f)^{N-r}
$$

The mean is:
$$
    \Epsilon[r] \equiv \sum_{r=0}^N P(r | f, N) r
$$

The variance is:
$$
\begin{align*}
    \text{var}[r] &\equiv \Epsilon[(r - \Epsilon[r])^2] \\
    &= \Epsilon[r^r] - (\Epsilon[r])^2 \\
    &= \sum_{r=0}^N P(r | f, N) r^2 - (\Epsilon[r])^2
\end{align*}
$$

Computing these sums are cumbersome. It is easier to observe that $r$ (the number of heads) is the sum of $N$ independent random variables - each random variable is whether the $i$th toss results in a head.

In general, for any random variables $x$ and $y$, if $x$ and $y$ are independent:
$$
\begin{align*}
    \Epsilon[x + y] &= \Epsilon[x] + \Epsilon[y]\\
    \text{var}[x + y] &= \text{var}(x) + \text{var}(y)
\end{align*}
$$

So the mean:
$$
\begin{align*}
    \Epsilon[r] &= N \cdot [f \times 1 + (1-f) \times 0] \\
    &= Nf
\end{align*}
$$

And the variance (note that we subtract $f^2$ as the expectation of a single trial is $f$):
$$
\begin{align*}
    \text{var}[r] &= N \cdot [f \times 1^2 + (1-f) \times 0^2 - f^2] \\
    &= N f (1-f)
\end{align*}
$$

## Stirling's Approximation

We will use the poisson distribution and the gaussian approximation to derive stirling's approximation for the factorial function.

Start with a poisson distribution with mean $\lambda$:
$$
    P(r | \lambda) = e^{-\lambda} \frac{\lambda^r}{r!} \ \ r \in \{ 0, 1, 2, ... \}
$$

For large $\lambda$, the poisson distribution is well approximated in the vicinity of $r \approx \lambda$ by a gaussian distribution with mean $\lambda$ and variance $\lambda$:
$$
    e^{-\lambda} \frac{\lambda^r}{r!} \approx \frac{1}{\sqrt{2 \pi \lambda}} e^{-\frac{(r-\lambda)^2}{2 \lambda}}
$$

Plugging $r = \lambda$, we get:
$$
\begin{align*}
    e^{-\lambda} \frac{\lambda^\lambda}{\lambda!} &\approx \frac{1}{\sqrt{2 \pi \lambda}} \\
    \lambda ! &\approx \lambda^\lambda e^{-\lambda} \sqrt{2 \pi \lambda}
\end{align*}
$$

Applying $\ln$, we get stirling's approximation for the factorial function:
$$
\begin{align*}
    x! &\approx x^x e^{-x} \sqrt{2 \pi x}\\
    \ln x! &\approx x \ln x - x + \frac{1}{2} \ln 2 \pi x
\end{align*}
$$

Now we apply stirling's approximation to $\ln \binom{N}{r}$ and re-organize:
$$
\begin{align*}
    \ln \binom{N}{r} &\equiv \ln \frac{
        N!
    }{
        (N-r)!r!
    } \\
    &\approx N \ln N - (N-r) \ln (N-r) - r \ln r \\
    &\approx N \ln \frac{N}{N - r} - r \ln \frac{1}{N-r} + r \ln \frac{1}{r} \\
    &\approx (N-r) \ln \frac{N}{N-r} + r \ln \frac{N}{r}
\end{align*}
$$

Note that:
- We are using the approximation of $\ln x! \approx x \ln x - x$
- In the second line, we replace all the factorials with the approximation, and notice that the $-x$ part of the approximation will cancel out between all the terms
- In the third line, we group terms together and use the trick $\ln x = -\ln\frac{1}{x}$ to flip signs
- In the last line, we artificially add and subtract $r \ln N$ to create the final form

Now, the whole point of this derivation is to write the approximation in a form that resembles binary entropy. We now define that concept here.

### Binary Entropy

The <<binary entropy function>> is ($\log$ denotes logarithm with base $2$):
$$
    H_2(x) \equiv x \log \frac{1}{x} + (1 - x) \log \frac{1}{1-x}
$$

> Note that the binary entropy function is just the special case of the general shannon entropy when we have only two possible outcomes, assuming $x$ represents a probability.

Let us rewrite $\log \binom{N}{r}$ using the binary entropy function:
$$
\begin{align*}
    \log \binom{N}{R} 
    &\approx N H_2(r / N)\\
    &= N \left( \frac{r}{N} \log \frac{N}{r} + \frac{N-r}{N} \log \frac{N}{N-r} \right)\\
    &= (N-r) \log \frac{N}{N-r} + r \log \frac{N}{r}
\end{align*}
$$

Or equivalently,
$$
    \binom{N}{r} \approx 2^{N H_2(r / N)}
$$

So we see that the binary entropy function nicely describes the combinatorial explosion of the binomial function. We can observe that since the binary entropy function is maximized when $\frac{r}{N} = 1/2$, the binomial combination is maximized likewise.

