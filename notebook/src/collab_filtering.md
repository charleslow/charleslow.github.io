# Collaborative Filtering

This post is trying to grok my colleague [Yuxuan's](https://www.linkedin.com/in/yxtay) sharing on various losses for collaborative filtering. All credit goes to him.

Collaborative filtering is typically done with implicit feedback in the RecSys setting. In this setting, interactions are often very sparse. Most of the time, only positive signals are recorded, but a non-interaction could either mean (i) user dislikes the item or (ii) the user was not exposed to the item. Hence, we cannot use algorithms like SVD which assume no interactions as irrelevance.

A generic and fairly common architecture for the collaborative filtering model is to embed each user and item into separate fixed size vectors, and use the cosine similarity between the vectors to represent a score. This score is fed into a cross entropy loss against the labelled relevance of user to item to train the embeddings.

## Setup

Let $f(u)$ and $f(i)$ denote the $k$ dimensional embedding vector for user $u$ and item $i$. Let the similarity function be $s(u,i)$ which is typically $f(u)^T f(i)$, and distance function $d(u, i)$ which is typically $||f(u) - f(i)||^2_2$. Then some common loss functions may be denoted as below.

<Pointwise Losses> are typically low-performing. For a given `(u, i)` pair, pointwise losses assume the presence of a `0, 1` label for relevance, and tries to predict it. The typical pointwise loss is Binary Cross Entropy, which may be expressed as:

$$
    \mathcal{L}_{BCE} = \sum_{(u,i) \in \mathcal{D}} \log \sigma (s(u,i)) - \sum_{(u,j) \notin \mathcal{D}} \log ( 1 - \sigma(s(u,j)) )
$$

<Pairwise Losses> assume the presence of training triplets `(u, i, j)` which correspond to user, positive item and negative item. A typical pairwise loss is Bayesian Personalized Ranking, as follows:

$$
    -\sum_{(u,i,j) \in \tau} \log \sigma \left[ 
        s(u, i) - s(u, j)
    \right]
$$


