# Van Den Oord 2017 - VQ-VAE

[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

This paper proposes a Variational Auto Encoder (VAE) that uses discrete (categorical) latent variables instead of continuous ones. Discrete latents are more desirable from an interpretability perspective, but historically have not been able to perform as well as continuous ones. This paper is apparently the first to bridge this performance gap.

We are in the domain of computer vision, where the VAE is used to generate new images from existing ones.

Note that although VQ-VAE starts with the VAE framework, it is not strictly variational because of the deterministic operations within. It is more precise to call it a (deterministic) autoencoder with vector quantization regularization. 

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

## Learning

At this point, we have fully specified the forward pass:
- Start with input $x$
- Encode into $z_e(x) \in \R^D$
- Use embedding table lookup to find nearest neighbour $z_q(x) = e_k$
- Decode into $p(x' | z_q(x))$

But it is not yet clear what the optimization objective and gradient flow should be. Recall that the standard VAE objective is to optimize: 
$$
    \log P(X | z) + \mathcal{D}_{KL}(Q(z|X) || P(z))
$$

The first term is the reconstruction loss, where we draw from the $Q$ distribution that maps an input $x \mapsto z$ (using the encoder plus some random noise). Then we decode the $z$ back into input space and try to make it similar to input $x$. The second term is the KL divergence which tries to regularize the $Q$ distribution to be as simple as possible.

However, we cannot simply use this equation for the VQ-VAE:
- While the reconstruction loss $\log P( X | z )$ is no issue (we can use the standard gaussian formulation), the second KL divergence term is a constant as we saw above. Hence doing this naively just reduces to a standard deterministic auto-encoder.
- Another problem is that in computing $P(X | z)$, we have to go through the non-linear operation of looking up the embedding table. This does not allow the gradient to flow back into the encoder.

Hence the authors need to re-design the optimization objective.

The <<first decision is to use the straight through estimator>> to circumvent the gradient issue. For the non-linear operation $f(z_e(x)) = z_q(x)$, we compute the forward pass normally (i.e. embedding table lookup) but simply pass the gradients through during backpropagation. This means that we approximate: 
$$
    \frac{dL}{d z_e(x)} \approx \frac{dL}{d z_q(x)}
$$

This allows the encoder to still receive gradient updates despite the non-differentiable operation. The theoretical justification for this operation is given in an earlier Bengio 2013 paper. Intuitively, if $z_e(x)$ is close to $z_q(x)$, the gradients should still be meaningful.

The <<second decision is to use l2 distance to learn the embedding table>>. This is a form of dictionary learning. Specifically, we add a term to the loss:
$$
    || sg[z_e(x)] - e ||^2_2
$$

Note that:
- $e$ here refers to the closest embedding to a given $z_e(x)$. We want embeddings in the codebook to move toward the average encoded representation
- $sg[]$ is the stop gradient operation (e.g. `.detach()` in pytorch). It uses the value of $z_e(x)$ but does not pass gradients back to the encoder. Since the objective of this loss term is to learn the codebook, we do not wish to pass gradients back to the encoder

The <<third decision is to add a commitment loss to bound the encoder outputs>>. This part feels a bit more arbitrary. The authors say that with just the first two terms, there is nothing that tethers the encoder output, which can grow arbitrarily and perpetually be far away from the codebook embeddings. The solution is to include the reverse direction from the dictionary learning loss:
$$
    \beta || z_e(x) - sg[e] ||^2_2
$$

Notice that this is identical to the second term except that the stop gradient operator is applied to the codebook embedding. Thus this gradient pushes the encoder embedding to be closer to its nearest codebook embedding. $\beta$ is a hyperparameter but the authors observed that results were robust to a wide range of $\beta$ values (`0.1`to `2.0`).

> A natural question is to wonder why we need both the second and third term which are identical except for where the stop gradient is placed. Why can't we just do $|| z_e(x) - e ||^2_2$ in a single term?
> 
> It appears that this will result in unstable training, because both sides (encoder and codebook embeddings) are simultaneously moving. This is a common issue in training things like GANs. Separating the terms results in more stable training.

> There is also a close connection between VQ-VAE and k-means clustering (as a special case of expectation maximization). The step of assigning each encoder output to the nearest codebook embedding is akin to assign a cluster for each data point in k-means. The step of updating the codebook embedding is akin to updating the cluster centroids in k-means. This idea is explored in subsequent papers like [Roy 2018](https://arxiv.org/abs/1805.11063).

## Learning the Prior Distribution

Up to this point, we assumed that the prior $P(z)$ is a uniform categorical.  