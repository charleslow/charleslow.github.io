# Weinberger 2009 - Hashing for Multitask Learning

[Weinberger 2009 - Feature Hashing for Large Scale Multitask Learning
](https://arxiv.org/abs/0902.2206)

This paper proposes a method to represent a large feature set in a smaller space by using hashing. It shows analytically that with a sufficient hash dimension $M$:
- The inner product between instances is preserved, i.e. doing a dot product between instances in the hashed dimension approximates the true dot product
- The same applies to learning a weight vector to generate predictions in the hashed space: the error of approximation goes to zero as $M$ increases

## Setup

Consider having data points $x^{(1)}, ..., x^{(n)} \in \R^d$, where $d$ can be very large (e.g. millions). This setting is easily realized when we use, for example, word bi-grams and tri-grams as features to perform some kind of text classification task. Such a large feature vector is unwieldy, and also inefficient since the feature vector is very sparse for a given text.

The <hashing trick> maps the high dimensional input vector to a smaller dimension feature space with the notation $\phi: \mathcal{X} \rightarrow \R^m$, such that $m << d$.

We start with the following definitions:
- Let $h$ be a hash function $h: \mathbb{N} \rightarrow \{1, ..., m \}$
- Let $\mathcal{E}$ be a hash function $\mathcal{E}: \mathbb{N} \rightarrow \{ \pm 1 \}$

Note that while the definitions map from an input integer, we may apply them to texts as well, since any finite-length text may be assigned to a unique integer. This is typically done in practice by applying some hash algorithm to a given string, and then using the modulo function to restrict it to the desired range.

With this, and given two vectors $x, x' \in \R^d$, we define the hash feature map:

$$
    \phi_j^{(h, \mathcal{E})}(x) = \sum_{i \in \mathbb{Z}\ :\  h(i)=j,\ 1 \leq i \leq d } \mathcal{E}(i)x_i
$$

Where $j \in 1, ..., m$ is an index in the hashed dimension space, and $i \in 1, ..., d$ is an index in the input dimension space. We get a hash collision if more than one $i$ term is hashed into a given position $j$. For brevity, we may just write $\phi_j^{(h,\mathcal{E})} := \phi$.

## Analysis

With this setup, the paper aims to prove analytically that hashing in this manner preserves the characteristics of the original space. In other words, we can significantly reduce the dimension of our features but achieve the same predictive effect as the original space by doing the hashing trick. This also means that the detrimental effect of hash collisions is minimal with a sufficiently large $m$.

We won't trace through all the results, just the important and simple ones.

### The hash kernel is unbiased

**Lemma 2** The hash kernel is unbiased, i.e. $\mathbb{E}_\phi \left[ \<x, x'\> \right]


