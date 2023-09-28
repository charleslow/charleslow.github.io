# Recommendations as Treatments

[Paper Link](https://arxiv.org/pdf/1602.05352.pdf).

Training and evaluation data for recommender systems is subject to selection bias, because the probability of observing a data point depends on (i) users interacting with items which they like and (ii) the recommender system pushing items which they think the user likes. This leads to data Missing Not at Random (MNAR) and leads to biased model training and evaluation.

Consider users interacting with movies. Denote users \\( u \in \\{ 1, ..., U \\} \\) and movies \\( i \in \\{ 1, ..., I \\} \\). Let \\( Y \in \mathbb{R}^{U \times I} \\) denote the true rating / interaction matrix between user and item, and \\( \hat{Y} \\) the predicted matrix. Let \\( O \in \\{ 0, 1 \\}^{U \times I} \\) be the observability matrix of whether an interaction was observed, and \\( P \in \mathbb{R}^{U \times I} \\) be the probability of observation, i.e. \\( P_{u,i} = P(O_{u,i} = 1) \\). Ideally, we want the propensity matrix `P` to be uniform across all entries, which gives us the Missing Completely at Random (MCAR) condition.

## Evaluating only on observed entries is biased

Given a model \\( \hat{Y} \\), we often want to compute evaluation metrics \\( \delta_{u,i}(\hat{Y}, Y) \\) on how well \\( \hat{Y} \\) approximates \\( Y \\). For example:

\\[
    \delta_{u,i}(\hat{Y}, Y) = (Y_{u,i} - \hat{Y}_{u,i})^2 \ \ \text{MSE}
\\]

\\[
    \delta_{u,i}(\hat{Y}, Y) = \frac{Y_{u,i}}{logrank( \hat{Y}_{u,i} )} \ \ \text{DCG}
\\]

The Risk function may then be denoted:

\\[
    R(\hat{Y}) = \frac{\sum_u \sum_i \delta_{u,i}(\hat{Y}, Y)}{U \times I}
\\]

However, \\( R(\hat{Y}) \\) cannot be computed because most entries in \\( Y \\) are missing. Typically, we estimate \\( \hat{R} \\) using only observed entries. This estimator is naive because it simply assumes the propensity matrix \\( P \\) is uniform. This assumption is often false in recommender systems because a small set of popular items tend to get impressed a lot. Hence this estimator will favour models that lean towards the popular items compared to models that recommend rarer / new items.

\\[
    \hat{R}\_{\text{naive}}(\hat{Y}) = 
        \frac{1}{| \{ (u, i): O_{u,i} = 1  \} |} 
            \cdot 
        \sum_{u,i: O_{u,i}=1} \delta_{u,i}(\hat{Y}, Y)
\\]

## Unbiased estimators

The main idea in this paper is to view this problem as analogous to estimating treatment effects in causal inference. Think of recommending an item as an intervention analogous to treating a patient with a specific drug. The goal is to estimate the outcome of a new recommendation (clicked or not) or new treatment (recovered or not), while most outcomes between `u,i` pairs are not known.

The key to resolving this bias is to understand the assignment mechanism that generated \\( O \\), namely the propensity matrix \\( P \\). We can then correct for the propensity. We need to assume that \\( P_{u,i} > 0 \forall u, i \\), i.e. full support, because otherwise the IPS estimator below is undefined.

## IPS Estimator

The main estimator is the Inverse Propensity Score (IPS) estimator. Assuming that the assignment mechanism \\( P \\) is known:

\\[
    \hat{R}\_{\text{ips}}(\hat{Y} | P) = 
        \frac{1}{| \{ (u, i): O_{u,i} = 1  \} |} 
            \cdot 
        \sum_{u,i: O_{u,i}=1} \frac{\delta_{u,i}(\hat{Y}, Y)}{P_{u,i}}
\\]

We have simply normalized each score by its inverse propensity, so that popular items with a high chance of being shown get their score reduced proportionally (and likewise in the opposite direction for rare items).

We can show that the IPS estimator is unbiased as we take the expectation over the random observability matrix. That is, suppose we are allowed to sample an infinitely large number of observations based on \\( P_{u,i} \\), the average of the IPS estimator over these datasets will be the true risk function. This simply happens because the expected value of \\( O_{u,i} \\) is simply the propensity, and if we know the propensity we can just cancel it out.

\\[
    \mathbb{E}_O [ \hat{R}\_{\text{ips}}(\hat{Y} | P) ]
    = 
        \frac{1}{U \cdot I} \sum_u \sum_i \mathbb{E}\_{O\_{u,i}}
            \left[ \frac{\delta\_{u,i}(\hat{Y}, Y)}{P\_{u,i}} \cdot O\_{u,i} \right]
    \\\\ =
        \frac{1}{U \cdot I} \sum_u \sum_i \delta\_{u,i}(\hat{Y}, Y)
    = R(\hat{Y})
\\]

## Comments

* This paper only addresses exposure bias, but not position bias. In other words, it assumes that all impressions are equal, which is valid when a user is shown an ad, but not when a user is shown a list of search results. In the latter case, items in lower positions have a lower probability of being considered by the user, even though both are considered to be observed. 