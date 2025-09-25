# Doersch 2016 - Tutorial on VAEs

[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)

Tutorial on VAEs from a computer vision perspective.

## Introduction

Generative modelling deals with models of distributions $P(X)$, defined over data points $X \in \mathcal{X}$. For example, $X$ may represent pixels of an image. $P(X)$ may indicate that a set of pixels that look like a real image has high probability, whereas images looking like random noise get low probability.

For generative modelling to be useful, we don't just want an unconditional distribution. Instead, we want to generate images from a conditional distribution, e.g. images that look like another image or following a certain caption.

## Latent Variable Models

Generative modelling usually starts with a latent variable. For example, in digit image generation, we may first want to choose a particular digit (say `5`) to generate. The image model then knows to generate pixels corresponding to this digit (as opposed to deciding on the fly). $z$ is typically chosen to indicate the latent variable (which can be multi-dimensional). 

Formally, we have a vector of latent variables $z \in \mathcal{Z}$ which we can easily sample from based on some probability density function $P(z)$ defined over $\mathcal{Z}$. In VAEs, this distribution is typically a multi-variate standard gaussian distribution. 

We now want to make sure that we have some function $f$ that maps from the latent variable space $\mathcal{Z}$ to the output space $\mathcal{X}$. Say we have a family of deterministic functions (typically some neural network) $f(z; \theta)$ which is parametrized by a vector $\theta \in \Theta$, where $f: \mathcal{Z} \times \Theta \rightarrow \mathcal{X}$. In this setup, randomness is injected by the random variable $z$ such that $f(z; \theta)$ is a random variable in the space $\mathcal{X}$.

To actually generate images that look like $X$, we wish to optimize $\theta$ such that we can sample $z$ from $P(z)$ and with high probability, $f(z;\theta)$ will look like that $X$'s in our data. 

How do we formalize this notion? Naively, $f(z; \theta)$ just gives a random point prediction. We need some probability distribution $P(X | z; \theta)$ which tells us how likely a given training sample $X$ is under the generative model $f$. By the law of total probability, we can then write down the Maximum Likelihood objective:
$$
\begin{align*}
    P(X) = \int P(X|z; \theta) P(z) dz
\end{align*}
$$

The choice of $P(X|z; \theta)$ for VAEs is often <<gaussian>>. Specifically, the most standard choice is:
$$
    P(X|z; \theta) = \mathcal{N}(X | f(z; \theta), \sigma^2 \times I)
$$

This means that the function $f(z; \theta)$ gives us the gaussian mean, and the covariance matrix is a fixed diagonal (as $\sigma$ is a hyperparameter). Notice that this gaussian expression allows us to express the idea that we just need $f(z; \theta)$ to produce samples that *look like* $X$, but does not have to exactly match some $X$.

This smoothness is critical for generative modelling &mdash; if we specify $P(X|z; \theta)$ to be the dirac delta function (i.e. all the probability mass is on the specific output produced by $f(z; \theta)$), it would be impossible to learn from data, as the function is zero almost everywhere.

Also note that while gaussian is the most common choice, it does not have to be so. We just need $P(X|z)$ to be computable and be continuous in $\theta$. For example, if $X$ is binary, then $P(X|z)$ can be a bernoulli parametrized by $f(z;\theta)$ (although it's unfathomable why one would do this).

## Variational Autoencoders

In order to maximize $P(X)$ above, there are two big problems to solve:
- How do we define the latent variables $z$ and what information they encode
- How to deal with the integral over $z$


### Problem 1: How to represent Latents?

For the first problem, VAEs basically prescribe minimal structure to the latents $z$, and say that the samples of $z$ can be drawn from a simple distribution, say $\mathcal{N}(0, I)$. Then, the onus falls on the model $f$ to map from a simple gaussian distribution to the complex distribution which describes our data. We know empirically that this is not a problem for an arbitrarily large neural network.

### Problem 2: How to deal with integral over P(z)?

For the second problem, the naive approach is to deal with the integral via <<sampling>>. That is we sample a large number of latents $z = \{ z_1, ..., z_n \}$ and compute $P(X) \approx \frac{1}{n} \sum_i P(X|z_i)$. The problem with this approach is that in high dimensional spaces, we need a very large $n$ to get an accurate estimate of $P(X)$. This is because for most instances of $z$, $P(X|z)$ will be very close to $0$.

The key idea behind VAEs is to <<speed up sampling by attempting to sample values of $z$ that are likely>> to have produced $X$, and compute $P(X)$ just from those. To do this, we need a new function $Q(z|X)$ which when given an image $X$, produces a distribution over $z$ values likely to have produced $X$. If the space of $z$ values that are likely under $Q$ is smaller than the space of $z$ values that are likely under $P(z)$, we will be able to estimate $E_{z \sim Q} P(X|z)$ much more cheaply.

However, by introducing a new arbitrary distribution $Q$, we are no longer sampling under $P(z)$, so we cannot directly obtain $P(X)$ from it. Thus we need some way to relate $E_{z \sim Q} P(X|z)$ and $P(X)$. This relationship is one of the cornerstones of variational Bayesian methods.

Let us start by defining <<KL divergence>> between $P(z|X)$ and some arbitrary distribution $Q(z)$ (which may or may not depend on $X$).
$$
    \mathcal{D}[Q(z) || P(z|X)] = E_{z \sim Q}[\log Q(z) - \log P(z | X)]
$$

Recall that KL divergence is asymmetric, and measures how different two probability distributions are. In this case, the expectation is taken over the distribution of $z$ values under $Q$.

Now apply Bayes rule to $P(z | X)$, and do some rearranging:  
$$
\begin{align*}
    \mathcal{D}[Q(z) || P(z|X)] &= E_{z \sim Q}[\log Q(z) - \log P(X | z) - \log P(z)] + \log P(X)\\
    \log P(X) - \mathcal{D}[Q(z) || P(z|X)] &= E_{z \sim Q}[\log P(X |z)] - \mathcal{D}[Q(z) || P(z)]\\
\end{align*}
$$

Some comments on the above:
- $P(X)$ comes out as a constant because it does not depend on $z$.
- We can group $Q(z)$ and $P(z)$ together into its own KL divergence term.

So far, we have not made any assumption on the arbitrary distribution $Q(z)$. In the context of trying to maximize $P(X)$, it makes sense to construct a $Q$ which does depend on $X$. So we make that dependency on $X$ explicit. Let's call this the <<ELBO equation>>. 
$$
\begin{align*}
    \log P(X) - \mathcal{D}[Q(z | X) || P(z|X)] &= E_{z \sim Q}[\log P(X |z)] - \mathcal{D}[Q(z | X) || P(z)]\\
\end{align*}
$$

This equation is core to the VAE, so we should understand it deeply.
- The left hand side represents what we want to optimize:
    - $\log P(X)$ was the original maximum likelihood objective - we want our model to produce images that look like $X$
    - $\mathcal{D}[Q(z|X) || P(z | X)]$ is the error or deviation of our tractable, estimated distribution $Q(z | X)$ from the true, intractable oracle distribution $P(z | X)$. This term is always more than or equals to $0$ and is $0$ if and only if $Q = P$.
- The right hand side is called the <<Evidence Lower Bound (ELBO)>>. In bayesian statistics, the marginal $P(X)$ is called the evidence, because our data $X$ is evidence for how good our model is. The RHS is a lower bound for our evidence precisely because the KL divergence term $\mathcal{D}[Q(z|X) || P(z|X)] \geq 0$, which implies $\log P(X) \geq \text{RHS}$.
- We cannot directly optimize $\log P(X)$, but we can do the next best thing which is to optimize the tractable RHS, given an appropriate choice of $Q$.
- As we increase the capacity of $Q$, the "error" term should become smaller and smaller, so the RHS will more accurately estimate the evidence (and lead to better optimization)

Notice how the RHS now resembles an auto-encoder:
- $Q$ "encodes" $X$ into a latent $z$
- $P$ "decodes" $z$ back to reconstruct $X$

### Optimizing the Objective

Now we need to perform SGD on the RHS. First we need to specify $Q(z|X)$. The usual choice is:
$$
    Q(z | X) = \mathcal{N}(z | \mu(X; \theta), \Sigma(X; \theta))
$$

Where $\mu$ and $\Sigma$ are both neural networks with parameters $\theta$ that map a given $X$ deterministically into a mean and variance vector respectively. We only need a vector for the variance because $\Sigma$ is typically constrained to be a diagonal matrix.

Because we chose both $Q(z|X), P(z)$ to be multi-variate gaussians, the KL divergence between them may now be computed in closed form ($k$ is the dimensionality of the distribution):

$$
\begin{align*}
    
    &\mathcal{D}[\mathcal{N}(\mu_0, \Sigma_0) || \mathcal{N}(\mu_1, \Sigma_1)]\\
    &= \frac{1}{2}
    \left(
        \text{tr}(\Sigma_1^{-1} \Sigma_0) + 
        (\mu_1 - \mu_0)^\intercal \Sigma_1^{-1}(\mu_1 - \mu_0) - k +
        \log(\frac{\det \Sigma_1}{\det \Sigma_0})
    \right)\\
    &= \frac{1}{2}
    \left(
        \text{tr}(\Sigma(X)) + 
        \mu(X)^\intercal \mu(X) - k +
        \log \det (\Sigma(X))
    \right)
\end{align*}
$$

The second line above gives the general case, but since our $P(z) = \mathcal{N}(0, I)$, it reduces into the third line. The functions $\mu(X), \Sigma(X)$ express the fact that the parameters of the normal distribution of $Q$ are determined by $X$. 

### Reparametrization Trick

So we have the second term on the RHS expressed as a function of $X$, which can be optimized via SGD. What about the first term $E_{z \sim Q}[\log P(X|z)]$?

This term is more tricky, because it involves two steps. Suppose we approximate the expectation by performing SGD. Then we have to:
- Sample a $z \sim Q$
- Compute $\log P(X | z)$ using the decoder

The first sampling step is not an operation that can be backpropagated through, so we cannot optimize this equation as-is. This is where the re-parametrization trick comes into play. 

The <<re-parametrization trick>> essentially moves the stochasticity of the sampling step out of the model forward pass into the data layer. Instead of sampling $z \sim Q$ directly, we first sample an intermediate $\epsilon \sim \mathcal{N}(0, I)$ and treat it as data. Then, we compute deterministically $z = \mu(X) + \Sigma^{1/2}(X) \times \epsilon$. This achieves the same effect as the direct sampling approach, but the difference is that the parameters of $Q$ have entered the equation in a deterministic way that can be backpropagated.

With the trick, we now have fully specified the optimization objective. The equation we take the gradient of is:
$$
E_{X \sim D} \left[
    E_{\epsilon \sim \mathcal{N}(0, I)} [
        \log P(X | z = \mu(X) + \Sigma^{1/2}(X) \times \epsilon)
    ]
    - \mathcal{D} [ Q(z|X) || P(z)]
\right]
$$

### Test Time Inference

At test time, when we want to generate new samples, we simply sample a new $z \sim \mathcal{N}(0, I)$, then feed it into the decoder to get a new $X = f(z)$. 

Suppose we want to evaluate the probability of this new test example, i.e. $P(X)$. This is not tractable for the reasons we discussed earlier, as it involves an integral over the distribution of $z$. However, we can make use of the ELBO concept and use the RHS of the ELBO equation as an approximation to $P(X)$. There is still an expectation over $z$, but because sampling $z \sim Q$ gives an expectation that converges much faster than sampling $z \sim \mathcal{N}(0, I)$, we can get a good sense of the probability by sampling a few times.

## Extra Info

This section tackles 3 questions to help our understanding:
1. How much error is introduced by the additional term $\mathcal{D}[Q(z|X) || P(z | X)]$?
2. How is the VAE framework linked to Minimum Description Length?
3. Do VAEs have regularization parameters analogous to sparsity penalties?

### Q1. Error from Lower Bound

Given that we are optimizing for the RHS and not directly for $\log P(X)$, how much error does the additional term $\mathcal{D}[Q(z|X) || P(z | X)]$ introduce? 

Since we assumed that $Q(z | X)$ takes the form of a high dimension gaussian, $P(z | X)$ must also take the form of a gaussian for the KL divergence term to go to $0$. However, this is not necessarily the case - we make no assumption on the distribution of $P(z | X)$. The hope is that if $f$ is sufficiently high-capacity, then there exists some $f$ which both (i) maximizes $P(X)$ and (ii) results in a gaussian-like $P(z | X)$. If such a function exists, then our objective would find it due to the $\mathcal{D}[Q(z|X) || P(z | X)]$ term.

### Q2. Minimum Description Length interpretation

(I don't understand this part.)

Another way to look at the RHS of the ELBO equation is in terms of information theory. $\log P(X)$ may be seen as the total number of bits required to construct a given $X$ under our model using an ideal encoding. The RHS views this as a two-step process to construct $X$.
- We first use some bits to construct $z$. $\mathcal{D}[Q(z|X) || P(z)]$ may be viewed as the expected information required to convert an uninformative sample from $P(z)$ to a sample from $Q(z|X)$
- In the second step, $P(X | z)$ measures the amount of information required to reconstruct $X$ from $z$ under an ideal encoding.

Hence the total number of bits $-\log P(X)$ is the sum of these two steps, minus a penalty we pay for $Q$ being a sub-optimal encoding $\mathcal{D}[Q(z|X) || P(z | X)]$.

### Q3. Regularization Effect

It is interesting to view the $\mathcal{D}[Q(z | X) || P(z)]$ term as regularization, since it is encouraging our $Q$ distribution to be similar to a simple distribution. 

In a usual sparse autoencoder, we have a $\lambda$ parameter in an objective function:
$$
    || \phi(\psi(X)) - X ||^2 + \lambda || \psi(X) ||_0
$$

That is, for encoder $\psi$ and decoder $\phi$, we encourage the encoding to be sparse. Similarly, the KL divergence term encourages our encoder $Q$ to be simple.

Where does a similar parameter like $\lambda$ enter into the ELBO equation? Recall that we chose a normal distribution for $P(X | z) \sim \mathcal{N}(f(z), \sigma^2 \times I)$. It turns out that $\sigma$ plays a similar role to $\lambda$, as we shall see.

Using the PDF of the normal distribution, we have that $\log P(X | z) = C - \frac{1}{2} || X - f(z) ||^2 / \sigma^2$, where $C$ is a constant that does not depend on $f$ and can be ignored during optimization. In the ELBO equation, $\sigma$ appears in the first term of the RHS but not the second term. Hence, by varying $\sigma$, we can control the relative weighting between the two terms. Specifically, a lower $\sigma$ implies less regularization and a larger $\sigma$ implies more regularization.

## Conditional VAEs






