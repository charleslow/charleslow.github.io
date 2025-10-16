# Van Den Oord 2017 - VQ-VAE

This paper proposes a Variational Auto Encoder (VAE) that uses discrete (categorical) latent variables instead of continuous ones. Discrete latents are more desirable from an interpretability perspective, but historically have not been able to perform as well as continuous ones. This paper is apparently the first to bridge this performance gap.

We are in the domain of computer vision, where the VAE is used to generate new images from existing ones.

## VAEs

A quick recap on regular VAEs:
- We start with some input $x$ which is an image
- The encoder encodes the input using $q(z | x)$ into latent space
- Where $z$ in this paper is a discrete latent variable
- $P(z)$ is the prior distribution (a uniform categorical in this paper)
- $P(x | z)$ is the decoder that decodes a given latent back into input space (i.e. generates an image)

## VQ-VAE

In VQ-VAE, the prior and posterior distributions are categorical. Drawing from the categorical distribution gives us an index $1, ..., K$ which is used to index into an embedding table comprising $D$ dimensional embeddings. This extracted embedding is then used to represent the sample and fed into the decoder model.

More specifically, let us define a latent embedding space $e \in \R^{K \times D}$. Starting with an input $x$, we pass it into the encoder to produce $z_e(x)$, which is a $D$-dimensional vector. We then find the nearest embedding to $z_e(x)$ in the embedding table to get a categorical index $k$ and a corresponding embedding $e_k$.

One way to think about this operation is that we are applying a particular non-linear operation that maps $z_e(x) \mapsto e_k$. Noticeably, this non linear operation is non-differentiable, which we will need to tackle later on.

We can thus define the posterior probability distribution for $q(z | x)$ as:
$$
    q(z = k | x) = \begin{cases}
        1   & \text{if } k = \argmin_j || z_e(x) - e_j ||_2\\
        0   & \text{otherwise}
    \end{cases}
$$ 

Note that this posterior distribution is deterministic. If we define a simple uniform prior for $p(z)$, we get that the KL divergence is constant: $D_{KL}(q || p) = \log K$.

Recall that:
$$
    D_{KL}(Q || P) = \sum_i Q(i) \log \frac{Q(i)}{P(i)}
$$

Since $Q(i) = 1$ if $i = k$ and $0$ otherwise, only one term in the summation is non-zero:
$$
\begin{align*}
    D_{KL}(Q || P) &= 1 \cdot \log \frac{1}{1 / K}\\
    &= \log K
\end{align*}
$$
