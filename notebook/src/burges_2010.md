# From RankNET $\rightarrow$ LambdaRank $\rightarrow$ LambdaMART

[Paper Link](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf).

This is an overview paper that explains the model behind LambdaMART, a technique used to learn Gradient Boosted Trees that optimize for NDCG on recommendation-type of datasets.

## RankNET

RankNET is a generic framework for training a learn to rank model. Given a differentiable model $ f : \mathbb{R}^d \mapsto \mathbb{R} $ that produces a score from an n-dimensional input vector, RankNET is able to train $f$ such that for a given query session with items $i=1,...n$ and corresponding features $x_i \in \mathbb{R}^d: i=1, ..., n$, $f$ learns to predict the relationship $f(x_i) > f(x_j)$ for any two items $i, j$ in the query session when $i$ is a better recommendation than $j$. The differentiable model $ f $ is typically a neural network or boosted trees.

RankNET uses revealed pair-wise preferences within each query session to train $f$. Specifically, suppose we have a data for one query session as follows:

| item  | clicked   |
| ----- | -------   |
| a     | 0         |
| b     | 1         |
| c     | 0         |

We can transform this data into a pairwise dataset as follows, where $ y_{jk} $ denotes the preference relationship between $item^j$ and $item^k$ which we inferred from the click data.

| $item^j$  | $item^k$ | $y^{jk}$|
| ----- | ----- | ----- |
| a     | b | -1 |
| a     | c | 0  |
| b     | a | 1 |
| b     | c | 0 |

The pairwise setting is now more amenable to modelling (compared to directly optimizing for a good ranking), since we can now treat the task as a classification problem. For each row of the pairwise dataset, we only need to model the probability that $item^j$ is preferred (or not) to $item^k$. This can be formalized using a cross entropy loss comparing the predicted preference of our model to the revealed preference in the dataset.

First, we model the predicted probability from the model. Given row $i$ of the pairwise dataset and $item^j$ and $item^k$ respectively, we model the predicted probability that $item^j$ is preferred to $item^k$ (using $\triangleright$ to denote a preference relationship) by passing the score difference between the predicted scores $\hat{y}^j_i := f(x^j_i)$ and $\hat{y}^k_i := f(x^k_i)$ for items j and k respectively through a sigmoid function, like so:

$$
    \hat{P}^{jk}_i := \hat{P}(item^j_i \ \triangleright \ item^k_i) = \frac{1}{1 + exp(-a (\hat{y}^j_i - \hat{y}^k_i))}
$$

Now let us denote the revealed probability that $item^j$ is preferred to $item^k$ as $P^{jk}_i$ such that:
- $P^{jk}_i = 1$ if we prefer item j to item k
- $P^{jk}_i = 0.5$ if we have no preference between the two items
- $P^{jk}_i = 0$ if we prefer item k to item j

The cross entropy loss of our model can then be expressed as:

$$
    L = \sum_i 
    \left[ 
        -P^{jk}_i log \hat{P}^{jk}_i - (1-P^{jk}_i) log (1-\hat{P}^{jk}_i)
    \right]
$$

For convenience, let us denote $y^{jk}_i$, such that $P^{jk}_i := \frac{1}{2} (1 + y^{jk}_i)$. Note that this translates into the following:
- $y^{jk}_i = 1$ if we prefer item j to item k
- $y^{jk}_i = 0$ if we have no preference between the two items
- $y^{jk}_i = -1$ if we prefer item k to item j

Let us also define the convenience variable $z = -log \hat{P}^{jk}_i = log \left[ 1 + exp(-a(\hat{y}^j_i - \hat{y}^k_i)) \right] $. The cross entropy loss then simplifies to (TBC):
$$
\begin{aligned}
    L 
    &= \sum_i 
    \left[ 
        -P^{jk}_i log \hat{P}^{jk}_i - (1-P^{jk}_i) log (1-\hat{P}^{jk}_i)
    \right]
        \\
    &= \sum_i
    \left[
        \frac{1}{2} (1 + y^{jk}_i) z 
        +
        -\frac{1}{2}
    \right]
\end{aligned}
$$

