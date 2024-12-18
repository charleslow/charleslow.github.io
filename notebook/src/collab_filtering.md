# Collaborative Filtering

This post is trying to grok my colleague [Yuxuan's](https://www.linkedin.com/in/yxtay) sharing on various losses for collaborative filtering. All credit goes to him.

Collaborative filtering is typically done with implicit feedback in the RecSys setting. In this setting, interactions are often very sparse. Most of the time, only positive signals are recorded, but a non-interaction could either mean (i) user dislikes the item or (ii) the user was not exposed to the item. Hence, we cannot use algorithms like SVD which assume no interactions as irrelevance.

A generic and fairly common architecture for the collaborative filtering model is to embed each user and item into separate fixed size vectors, and use the cosine similarity between the vectors to represent a score. This score is fed into a cross entropy loss against the labelled relevance of user to item to train the embeddings.

## Setup

Let $f(u)$ and $f(i)$ denote the $k$ dimensional embedding vector for user $u$ and item $i$. Let the similarity function be $s(u,i)$ which is typically $f(u)^T f(i)$, and distance function $d(u, i)$ which is typically $||f(u) - f(i)||^2_2$. Then some common loss functions may be denoted as below.

<<Pointwise Losses>> are typically low-performing. For a given `(u, i)` pair, pointwise losses assume the presence of a `0, 1` label for relevance, and tries to predict it. The typical pointwise loss is Binary Cross Entropy, which may be expressed as:

$$
    \mathcal{L}_{BCE} = \sum_{(u,i) \in \mathcal{D}} \log \sigma (s(u,i)) - \sum_{(u,j) \notin \mathcal{D}} \log ( 1 - \sigma(s(u,j)) )
$$

<<Pairwise Losses>> assume the presence of training triplets `(u, i, j)` which correspond to user, positive item and negative item. A typical pairwise loss is Bayesian Personalized Ranking, as follows:

$$
    -\sum_{(u,i,j) \in \tau} \log \sigma \left[ 
        s(u, i) - s(u, j)
    \right]
$$

## Weighted Matrix Factorization

This describes the [Cornac implementation](https://cornac.readthedocs.io/en/stable/api_ref/models.html#module-cornac.models.wmf.recom_wmf) of WMF. The code:
- [wmf.py](https://github.com/PreferredAI/cornac/blob/master/cornac/models/wmf/wmf.py)
- [recom_wmf.py](https://github.com/PreferredAI/cornac/blob/master/cornac/models/wmf/recom_wmf.py)

Let $A \in \R^{n \times m}$ describe a rating matrix of $n$ users and $m$ items. For simplicity, we may restrict $A_{ij} \in [0, 1]$. Given a user embedding matrix $U \in \R^{n \times k}$ and item embedding matrix $V \in \R^{m \times k}$, WMF computes the similarity score as the dot product $U \cdot V^T \in \R^{n \times m}$. 

The general loss function is:

$$
    \mathcal{L} = \sum_{i, j\ :\ A_{ij} = 1} \left(
        A_{ij} - U_i \cdot V_j^T
    \right)^2  + b \sum_{i ,j\ :\ A_{ij} = 0} \left(
        A_{ij} - U_i \cdot V_j^T
    \right)^2
    + u \cdot ||U||^2_F + v \cdot ||V||^2_F
$$

The idea is to simply take the squared error from the true ratings matrix as our loss, but apply a lower weightage to elements in the rating matrix where the rating is zero (as these are usually unobserved / implicit negatives that we are less confident about). Usually `b` is set to `0.01`. Regularization is performed on the user and item embedding matrices, with $u \in \R, v \in \R$ as hyperparameters to adjust the strength of regularization.

For cornac, this loss is adapted to the mini batch setting. Specifically, the algorithm is:
1. Draw a mini batch (default: `B = 128`) of items but use all the users
2. Compute the model predictions $P = U \cdot V_{\text{batch}}^T \in \R^{n \times B}$
3. Compute squared error $E = (A_{batch} - P)^2 \in \R^{n \times B}$
4. Multiply matrix of weights (either `1` for positive ratings or `b` for negative ratings) element-wise with $E$
5. $\text{loss} = \text{sum}(E) + u \cdot ||U||^2_F + v \cdot ||V_{batch}||^2_F$

Note that Adam optimizer is used, and gradients are clipped between `[-5, 5]`.

