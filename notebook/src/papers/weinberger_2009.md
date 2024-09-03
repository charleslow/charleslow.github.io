# Weinberger 2009 - Hashing for Multitask Learning

[Weinberger 2009 - Feature Hashing for Large Scale Multitask Learning
](https://arxiv.org/abs/0902.2206)

This paper proposes a method to represent a large feature set in a smaller space by using hashing. It shows analytically that with a sufficiently large hash dimension $m$:
- The inner product between instances is preserved, i.e. doing a dot product between instances in the hashed dimension approximates the true dot product in the original dimension
- The same applies to learning a weight vector to generate predictions in the hashed space: the error of approximation goes to zero as $M$ increases

## Setup

Consider having data points $x^{(1)}, ..., x^{(n)} \in \R^d$, where $d$ can be very large (e.g. millions). This setting is easily realized when we use, for example, word bi-grams and tri-grams as term-frequency features to perform some kind of text classification task. Such a large feature vector is unwieldy, and also inefficient since the feature vector is very sparse for a given text.

The <<hashing trick>> maps the high dimensional input vector to a smaller dimension feature space with the notation $\phi: \mathcal{X} \rightarrow \R^m$, such that $m << d$.

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


**Lemma 2** <<The hash kernel is unbiased>>, i.e. $\mathbb{E}_\phi \left[ \langle x, x' \rangle_\phi \right] = \langle x, x' \rangle$. 

**Proof**. The proof simply starts by expanding the inner product in the hashed space as follows:
$$
    \langle x, x' \rangle_\phi = \sum_{i=1,...,d} \sum_{j=1,..., d} 
        \mathcal{E}(i) \mathcal{E}(j) \cdot x_i x'_j \cdot \delta_{h(i), h(j)}
$$

Where $\delta_{h(i) = h(j)}$ is an indicator variable which takes $1$ if $h(i) = h(j)$ (i.e. they are hashed to the same position) and $0$ otherwise. 

To see that this expansion is true, consider a position in the hashed space, e.g. $k$. The value at position $k$ looks something like the following. We just need to move the summands to the left and use the $\delta$ variable to denote the common hash positions where $x_i$ and $x'_j$ interact (if `i` and `j` are hashed to different positions, they clearly do not interact in an inner product).
$$
    \left[ \langle x, x' \rangle_\phi \right]_k =
        \left[ 
            \sum_{i \in \mathbb{Z}\ :\  h(i)=k,\ 1 \leq i \leq d } \mathcal{E}(i)x_i
        \right]
        \left[ 
            \sum_{j \in \mathbb{Z}\ :\  h(j)=k,\ 1 \leq j \leq d } \mathcal{E}(j)x_j
        \right]
$$

Now note that we can decompose the expectation over $\phi$ into its independent constituents, i.e. $h$ and $\mathcal{E}$ respectively (since the two hashes are independent): 

$$
    \mathbb{E}_\phi \left[ \langle x, x' \rangle_\phi \right] = 
    \mathbb{E}_h \left[ 
        \mathbb{E}_{\mathcal{E}}
        \langle x, x' \rangle_\phi
    \right]
$$

Now we just need to observe that the hashed values $\mathcal{E}(i), \mathcal{E}(j)$ are independent from all other terms in general, but also independent from each other whenever $i \neq j$ (provided our hash function is pairwise independent). Thus when $i \neq j$, the summand is:

$$
    \mathbb{E}_{\mathcal{E}} \left[ \mathcal{E}(i) \right] \cdot
    \mathbb{E}_{\mathcal{E}} \left[ \mathcal{E}(j) \right] \cdot
    \mathbb{E}_{\mathcal{E}} \left[
        x_i x'_j \cdot \delta_{h(i), h(j)}
    \right]
$$

These are clearly $0$ because $\mathbb{E} \left[ \mathcal{E}(i) \right] = 0$. So the original summation reduces to:

$$
\begin{align*}
    \mathbb{E}_\phi \left[
        \langle x, x' \rangle_\phi
    \right] 
        &= \mathbb{E}_\phi \left[ 
            \sum_{i=1,...,d}
            \mathcal{E}(i)^2 \cdot x_i x'_i
        \right]\\
        &= \langle x, x' \rangle
\end{align*}
$$

Not only is the hashed inner product unbiased, it also has a <<variance that scales down in $O(\frac{1}{m})$>>. The proof does a similar but more tedious expansion as the above, and assumes that $x, x'$ have l2-norm of $1$. This suggests that the hashed inner product will be concentrated within $O(\frac{1}{\sqrt{m}})$ of the true value.

These results are sufficient to <<justify use of the hashed inner product space in practice>>. That is, we can perform recommendations in the hashed space with sufficiently large $m$ (we can tune that using validation error) to make the large feature space tractable. The paper goes on to prove more detailed bounds on the error and norm which are of less practical significance.

## Multi-task Learning

The authors argue that this method is especially useful in the multi-task learning setting. Consider an email spam classification task where the vocab space is $\mathcal{V}$ and the user space is $\mathcal{U}$. The parameter space is thus $\mathcal{V} \times \mathcal{U}$, i.e. we wish to learn a user-specific weight vector $w_u \in \R^{|\mathcal{V}|}$ for each user $u$, which allows us to personalize the spam filter for each user (different users have slightly differing definitions of what is spam).

The authors suggest the following approach:
- Use the hashing trick to hash each term $v$ into the hashed space. e.g. `data` is passed into a global hash function $\phi_0$ and assigned to a position
- Each user gets his/her own hash function $\phi_u$. This may be implemented by using the same hash function but appending the `user_id` like so: `user1_data`, which hashes the same term into a new position.
- We may thus represent each instance by $\phi_0(x) + \phi_u(x) \in \R^m$, capturing both a global element (some terms are universally spam-indicative) and a personalized element (some terms are specifically indicative for a user)
- Finally, we learn a weight parameter $w_h \in \R^m$ by training it in the hashed space

Empirically, for their task of $|\mathcal{V}|=\text{40 million}$, $|\mathcal{U}| = \text{400,000}$, they found that performance starts to saturate with $m \approx \text{4 million}$. This is a very small fraction of the total space $|\mathcal{V}| \times |\mathcal{U}|$, showing the effectiveness of their method. Nevertheless, we should note that `4 million` is still a rather large space.