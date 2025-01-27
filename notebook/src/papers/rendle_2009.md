# Rendle 2009 - Bayesian Personalized Ranking

[BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)

This is one of the most cited papers for collaborative filtering. It proposes a pairwise learning algorithm for item recommendation from implicit feedback that remains one of the most performant to date. It is pitched as a rival method to Weighted Matrix Factorization, the other strong model of the day. BPR argues that it is superior to WMF because it explicitly optimizes for ranking.

## Setup

Let $U$, $I$ denote the set of all users and items respectively. We have access to an implicit feedback data $S \subset U \times I$. The task is to produce each user with a personalized ranking $>_u \subset I^2$. Note that $I^2$ denotes the cartesian product of $I$ with itself, so it represents the set of all ordered pairs of elements of $I$. The ordering $>_u$ is a subset of these pairs where a preference relationship is indicated by the model. For convenience we also denote:
- $I_u^+ := \{ i \in I : (u, i) \in S\}$
- $U_i^+ := \{ u \in U : (u, i) \in S\}$


In implicit feedback systems, only positive interactions between user and item and observed. The typical approach is to put the interactions in a matrix and fill in unobserved entries with $0$. A model is then fitted to this model. The paper makes an interesting observation that this problem is ill-defined, since a perfectly expressive model that fits the training data perfectly will fail to make any prediction at all, since all unobserved entries will be given a score of $0$. Hence regularization methods are employed to avoid such overfitting.

The idea of BPR is to avoid making judgment on the pairwise preference between two items with the same score. That is, if items $i$ and $j$ are both interacted by user $u$, we cannot judge if one is preferred over the other. Also, if $i$ and $j$ are both unobserved interactions, we cannot make such judgement either. Thus the training data $D_S : I \times I \times I$ is denoted by the following. 

$$D_S := \{ (u, i, j) | i \in I_u^+ \text{ AND } j \in I \setminus I_u^+ \}$$

Note that this definition means that $(u, i, j) \in D_S \implies (u, j, i) \notin D_S$, since $j$ cannot be positive for $u$ if it was included as a negative.

## BPR Loss

The bayesian formulation for finding a ranking is to find the model parameters $\Theta$ that maximize the following probability:
$$
    p(\Theta | >_u) \propto p(>_u | \Theta) \cdot p(\Theta)
$$

We assume that:
- All users act independently of each other
- The ordering of each pair of items $(i, j)$ for a given user is independent of the ordering of every other pair

Then, we can write the likelihood across all users as:
$$
    \prod_{u \in U} p( >_u | \Theta ) = \prod_{(u, i, j) \in U \times I \times I}
        p(i >_u j | \Theta)^{\delta((u, i, j) \in D_S)} \cdot 
        \left[ 
            1 - p( i >_u j | \Theta)
        \right]^{\delta((u, j, i) \in D_S)}
$$
Where $\delta$ is the indicator function for the preference relationship. In other words, the likelihood of the overall ordering is the product of the likelihood of each $u, i, j$ triplet. For each $u, i, j$ triplet, the likelihood is given by the model's prediction for given label $0$ or $1$.

Note that the above term is only not equals to $1$ if $i >_u j$ or $j >_u i$. Also, since we observed above that $(u, i, j) \in D_S \implies (u, j, i) \notin D_S$ and vice versa, only one of the two terms will come into play for each entry. The above formula can be simplified to:
$$
    \prod_{u \in U} p( >_u | \Theta ) = \prod_{(u, i, j) \in D_S} p(i >_u j | \Theta)
$$

<<Note:>> Not too sure about the above step, would have thought that the $1-p$ term should also come into play when $j >_u i$.

## Model

Now we can model the preference probability using a model as:
$$
    p(i >_u j | \Theta) := \sigma{(\hat{y}_{uij}(\Theta))}
$$
Where $\hat_y$ denotes a BPR-agnostic model that generates a predicted real-valued score for the triplet $(u, i, j)$.

## Cornac Implementation

Cornac has a [Cython implementation of BPR](https://github.com/PreferredAI/cornac/blob/master/cornac/models/bpr/recom_bpr.pyx) that is fast (but not memory scalable when number of user and items is large). `num_threads` can be increased for faster parallel processing (Cython overrides the GIL).

At initialization:
- User embedding `U` of shape `n_users, k` is drawn randomly from a standard uniform distribution where `k` is the embedding dimension
- Then, we take $(U - 0.5) / k$. I suppose this leads to small, centered values which leads to stable initial training. Normalizing by `k` ensures that the dot product between vectors does not explode.
- Item embedding of shape `n_items, k` is initialized similarly
- Optional bias vector of shape `n_items` is initialized to the zero vector

Note that `train_set.X` is a `scipy.sparse.csr_matrix`, which has 3 main vectors:
- `data`: values of non-zero entries
- `indices`: column indices of the non-zero values
- `indptr`: index pointers to the start of each row in `data` and `indices`

The vector `user_ids` is constructed from the `csr_matrix`. It is of the same length as `X.indices` and represents the row indices of the non-zero values. Thus we have both row and column indices of the non-zero values.

The main training loop samples a `(u, i, j)` triplet randomly each turn. It does so by the following steps:
- A random index `i` between `0` and `len(user_ids)` is generated
- The `user_id` and `item_i_id` are obtained by indexing `user_ids[i]` and `X.indices[i]` respectively
- Now a random index `j` between `0` and `n_items` is generated
- The `item_j_id` is obtained by indexing `neg_item_ids[j]` where `neg_item_ids = np.arange(n_items)`
- A check is performed on the sparse matrix that `item_j_id` is not a positive item for `user_id`. If so, we skip this triplet.
- Pointers to the start of the relevant `u, i, j` embeddings are then obtained 
- Note that `1` epoch comprises `len(user_ids)` number of triplets

The Cython code then computes and manually applies the SGD updates as follows:
$$
\begin{align*}
    s &= U_u \cdot (I_i - I_j) + (B_i - B_j) \\
    z &= 1 / (1 + exp(s)) \\
    U_u &\pluseq \text{lr} \sum \left[ 
        z \times (I_i - I_j) - \lambda \times U_u
    \right] \\
\end{align*}
$$

Note that the prediction for the triplet is considered correct if $z < 0.5$.







