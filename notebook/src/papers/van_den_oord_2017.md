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

More specifically, let us define a latent embedding space $e \in \R^{K \times D}$. That is, we have $K$ discrete latents and $D$ is the embedding dimension of each latent. Starting with an input $x$, we pass it into the encoder to produce $z_e(x)$, which is a grid of $D$-dimensional vectors. We then find the nearest embedding to $z_e(x)$ in the embedding table to get a categorical index $k$ and a corresponding embedding $e_k$ for each encoded vector.

> <<Note>> that using our running example of image generation, $z_e(x)$ is encoded into a 2D grid of latents (say `32 x 32 x D`). We find the nearest embedding at each position such that we end up with a grid of `32 x 32` codebook indices. Since each position is discretized independently of the others, in the exposition we refer to $x, z_e(x)$ and so on as though it is one vector. 

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

Hence the KL divergence term is constant and ignored during training.

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

Hence, the final training objective becomes:
$$
    \L = \log p(x | z_q(x)) + || sg[z_e(x)] - e ||^2_2 + \beta || z_e(x) - sg[e]||^2_2
$$

## Evaluation

The log likelihood of the complete model can be evaluated as the total probability over all latents $k \in 1,...,N$. It is common practice to compute the log likelihood of a held out test set to evaluate how well our model has learned the data distribution:
$$
    \log p(x) = \log \sum_k p(x|z_k)p(z_k)
$$

Note that $-\log_2 p(x) / (H \times W \times \text{Channels})$ is also computed to report <<bits per dimension>>, which is a common way to evaluate such VAE models on test data. This is literally the number of bits required to represent our data under this model.

Because the decoder $p(x|z)$ is trained with $z = z_q(x)$ from MAP-inference, the decoder should end up placing all probability mass on $p(x|z_q(x))$ after full convergence and no probability mass on $p(x|z) \forall z \neq z_q(x)$. So we can write:
$$
    \log p(x) \approx \log p(x|z_q(x))p(z_q(x))
$$


## Learning the Prior Distribution

Up to this point, we assumed that the prior $P(z)$ is a uniform categorical for training the encoder, decoder and codebook embeddings. This may be viewed as a *training trick* or *mathematical convenience* to make our framework work. As you may recall, using the uniform prior resulted in a constant term for the KL-divergence, meaning that the term is ignored during training.

At inference time, when generating a new image, using a uniform prior will result in incoherent images. Instead, we need to train a separate decoder like a pixelCNN or transformer over the latent space to generate a coherent latent grid for decoding.

Specifically, we encode and quantize our training dataset into grids of `32 x 32` codebook indices. Then, an autoregressive decoder is trained to predict the codebook indices in a causal autoregressive way. For PixelCNN, it works on the 2D grid of `32 x 32`, but for transformer we need to linearize it in row-major form before training.

Now at inference time, we can start with a uniform latent index for the first position, then generate the subsequent positions using the latent decoder. After we have generated a grid of `32 x 32` latents, we look up embeddings to get `32 x 32 x D` grid. This is passed to the decoder to get the generated image.
- <<Note>> that since we are doing autoregressive generation in the compressed latent space, it is a lot faster than autoregressive generation in the original pixel or token space. This is the idea used in [Kaiser 2018 - Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382).