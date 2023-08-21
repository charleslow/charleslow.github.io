# Recommendations as Treatments

[Paper Link](https://arxiv.org/pdf/1602.05352.pdf).

Training and evaluation data for recommender systems is subject to selection bias, because the probability of observing a data point depends on (i) users interacting with items which they like and (ii) the recommender system pushing items which they think the user likes. This leads to data Missing Not at Random (MNAR) and leads to biased model training and evaluation.

Consider users interacting with movies. Denote users \\( u \in \\{ 1, ..., U \\} \\) and movies \\( i \in \\{ 1, ..., I \\} \\). Let \\( Y \in \mathbb{R}^{U \times I} \\) denote the true rating / interaction matrix between user and item, and \\( \hat{Y} \\) the predicted matrix. Let \\( O \in \\{ 0, 1 \\}^{U \times I} \\) be the observability matrix of whether an interaction was observed, and \\( P \in \mathbb{R}^{U \times I} \\) be the probability of observation, i.e. \\( P_{u,i} = P(O_{u,i} = 1) \\). Ideally, we want the propensity matrix `P` to be uncorrelated with labels `Y`.

## Evaluation

Given a model \\( \hat{Y} \\), we often want to compute evaluation metrics \\( \delta_{u,i}(\hat{Y}, Y) \\) on how well \\( \hat{Y} \\) approximates \\( Y \\). For example:

\\[
    \delta_{u,i}(\hat{Y}, Y) = (Y_{u,i} - \hat{y}_{u,i})^2 \ \ \text{MSE}
\\]

\\[
    \delta_{u,i}(\hat{Y}, Y) = \frac{Y_{u,i}}{logrank( \hat{y}_{u,i} )} \ \ \text{DCG}
\\]