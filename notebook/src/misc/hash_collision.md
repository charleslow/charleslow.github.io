# Hash Collisions

Bloom embeddings is one way of handling large number of user / item IDs. Instead of assigning each unique ID to a unique embedding, we may perform hashing to assign each unique ID to one or multiple bins. The embeddings at these positions are then taken and typically summed to get the final representation of each user / item.

One natural question arises: given `n` number of unique values and `m` number of bins, what is the expected number of bins to have 2 or more values assigned to it? This collision is especially a problem if we only represent each unique value with one embedding (i.e. `num_hashes = 1`). In practice, we would usually have at least `2` or more hashes to avoid this problem.

We may approach the problem by observing that each item is hashed uniformly and has a $\frac{1}{m}$ chance of landing in a particular bin. The hashing is also independent, i.e. it is not affected by the hashing for any other item.

This means that the expected number of items assigned to a particular bin may be modeled as a $X \sim \text{Binomial}(n, \frac{1}{m})$ distribution, since we have `n` trials and a fixed probability of "success" of `1/m`. 
- The probability we desire is $P(X \geq 2) = 1 - P(X=0) - P(X=1)$
- The PMF is $P(X = k) = \binom{n}{k} \left( \frac{1}{m} \right)^k \left( 1 - \frac{1}{m} \right)^{n-k}$, so:
    - $P(X=0) = \left( 1 - \frac{1}{m} \right)^n$
    - $P(X=1) = \frac{n}{m} \cdot \left( 1 - \frac{1}{m} \right)^{n-1}$

So we may put the above together to obtain $P(X \geq 2)$. It then remains to get the expected number of colliding bins by $m \cdot P(X \geq 2)$, by using the linearity of expectation.

For example, given $n=50,000, m=1,000,000$, the expected number of colliding bins is `1,210` (out of `1 million` bins), which is not insignificant.

Now note that since $n$ is large and $p = 1/m$ is small, $X$ may be well approximated by $Y \sim \text{Poisson}(n/m)$. Recall that the Poisson PMF is $P(Y = k) = \lambda^k \cdot e^{-\lambda} / k!$. So:
- $P(Y = 0) = e^{-n/m}$
- $P(Y = 1) = \frac{n}{m} \cdot e^{-n/m}$

Using this approximation, we get the expected number of colliding bins as `1,209`, which is a very good approximation. Hence the formula we want is:
$$
    \text{Expected bins with collision} =  m \times (1 - e^{-n/m} - \frac{n}{m} \cdot e^{-n/m})
$$