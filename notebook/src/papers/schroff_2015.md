# Schroff 2015 - FaceNET

[Schroff 2015 - FaceNet: A Unified Embedding for Face Recognition and Clustering](https://ar5iv.labs.arxiv.org/html/1503.03832)

This paper proposes the <<Triplet Loss>> for learning a face recognition system.

The task is that given a large number of person identities with a number of images associated with each person, to learn a model representation for an image in euclidean space, such that images belonging to the same person are close by and images for different persons are far away. This paper was impactful because it improved the SOTA on face verification by a large margin.

At this point, representation learning often trained a CNN classification model to classify images to a known identity. A bottleneck layer of relatively low dimensionality in the middle of the network is chosen as the representation of an image. In contrast to this indirect method, this paper directly optimizes the representation using <<contrastive learning>>.

## Setup

Let the embedding of image $x$ be represented by $f(x) \in \mathbb{R}^d$, and constrain $||f(x)||_2=1$. Now for a given person `i`, we want the anchor image $x_i^a$ to be closer to other images of the same person $x_i^p$ (positive) than images of any other person $x_i^n$ (negative), with some margin $\alpha$. That is, we desire a model $f$ such that:

$$
\begin{align*}
    &||f(x_i^a) - f(x_i^p)||^2_2 + \alpha < ||f(x_i^a) - f(x_i^n)||^2_2 \ \ \ \ &\forall (x_i^a, x_i^p, x_i^n) \in \Tau
\end{align*}
$$ 

Where $\Tau$ denotes the set of all possible triplets in the dataset. The authors note that most triplets would easily satisfy the desired constraint, hence we need some way to picking good triplets to optimize the learning process.

The loss we desire to minimize is thus as below. The intuition is that we wish to have small distances for the anchor-positive pair $d_{ap}$ but large distances for the anchor-negative pair $d_{an}$. 
- Easy triplets where $d_{an} - d_{ap} > \alpha$ will contribute 0 loss
- Semi-hard triplets where $0 < d_{an} - d_{ap} \leq \alpha$ contribute small losses
- Hard triplets where $d_{an} - d_{ap} < 0$ contribute large losses

$$
    \mathcal{L} = \sum_{i=1}^N \left[
        ||f(x_i^a) - f(x_i^p)||^2_2 \ - \ ||f(x_i^a) - f(x_i^n)||^2_2 + \alpha
    \right]_+
$$

## Triplet Selection

As with most contrastive learning methods, the authors observe that the selection of triplets is important for learning well. In an ideal world, we would want to select the hardest negatives for each anchor, i.e.: 
    $$x_i^n = argmin_{x_i^n} \ ||f(x_i^a) - f(x_i^n)||_2$$

However, this is undesirable because:
- We may end up mostly selecting mislabelled instances or corrupted images
- It is computationally infeasible to search across the whole corpus
- Trying to learn hard negatives at the start of training may cause learning to collapse

Hence, the authors propose:
- <<Mini batch negatives>>. Instead of finding the hardest negatives across the entire corpus, they mine for the hardest negatives in the mini-batch. This improves computational efficiency and mitigates the mislabelled/corrupted data issue.
- <<Curriculum Learning>>. At the start of training, the authors select easier triplets, and gradually increase the difficulty as training goes along. Presumably, this is done by thresholding negatives based on $d_{an} - d_{ap} > t$ and starting from high $t$ to low $t$.
- <<Semi-hard negatives>>. The authors mention that it is helpful to limit negatives to $d_{an} - d_{ap} > 0$, but it is unclear whether they always do this or only at the start of training. The intuition is to cap the difficulty of the triplets, as triplets that are *too* difficult may actually hinder learning.

## Application to Semantic Search

The triplet loss may be applied directly to semantic search. However, it is important to note that the paper assumes that for each person, <<all others-labelled instances are negatives>>. This is a suitable assumption for face recognition as each image can only belong to one person, but it is not true for semantic search, where a document may be relevant for multiple queries. Hence, the mislabelling issue when mining hard negatives is amplified for semantic search. I imagine that the selection of accurate negatives for semantic search would require more verification and filtering.

