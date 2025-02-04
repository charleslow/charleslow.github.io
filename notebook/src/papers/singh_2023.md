# Singh 2023 - Semantic IDs for Recs

[Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://arxiv.org/abs/2306.08121)

This paper proposes uses semantic IDs instead of hashing to random IDs to represent user or items. Semantic IDs mean that collisions to the same ID are semantically meaningful, and hence addresses the collision problem and also offers better generalization in cold start scenarios. The paper shows that this approach (i) retains memorization ability on par with random hashing and (ii) generalizes much better to unseen items and (iii) is much more computationally efficient as each user / item is represented simply by a `64-bit integer`.

## Background

In industry recommender systems, users or items are typically represented using random bloom hashing on a string ID (see [Weinberger 2009](./weinberger_2009.md)) into an embedding table. Such IDs have no semantic meaning, so collisions in this approach degrade performance. Nevertheless, it is empirically clear that the item-level memorization is important for good recommender performance. On the other extreme, one may choose to avoid IDs entirely and simply represent items using their content embedding. While it is clear that this obviates the collision problem and improves cold-start performance, multiple papers have shown that performance will be degraded overall due to inability to match the memorization ability of ID-based recommenders.

The ranker used in experiments is from Google's video recommender. See [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://arxiv.org/abs/2302.09178) and [Recommending What Video to Watch Next: A Multitask Ranking System](https://daiwk.github.io/assets/youtube-multitask.pdf).

## Approach

The approach pre-supposes that every item (video in this case) already has a good learned embedding. For Google's case, they have a video encoder that represents each YouTube video as a `2048` dimensional vector that captures the topicality of the video. The encoder takes both audio and visual features as input. The model training is described in [Large Scale Video Representation Learning via Relational Graph Clustering](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Large_Scale_Video_Representation_Learning_via_Relational_Graph_Clustering_CVPR_2020_paper.pdf).

The approach has two stages:
1. <<Represent each item as a sequence of Semantic IDs>>. They employ a method called RQ-VAE to compress a dense content embedding into a few discrete tokens (represented as integers) which capture the semantic content. 
2. <<Train the ranking model with Semantic IDs>>. Once trained, the RQ-VAE model is frozen and used to generate semantic IDs for each item. We then train embeddings for each semantic ID along with the rest of the ranking model.

## Stage 1: RQ-VAE for Semantic IDs

The idea of RQ-VAE is to iteratively find tokens that match the content embedding. Once a token is found, the next token will try to match the residual embedding and so on.

Let $x \in \R^d$ be the content embedding. The algorithm is as follows:
- Use encoder $\mathcal{E}$ to map the content embedding $x \in \R^D$ to a latent vector $z \in \R^{D'}$
- The residual quantizer of $L$ layers recursively quantizes $z$ into semantic IDs
    - Each level $l$ has a codebook $\mathcal{C}_l := \{ e^l_k \}^K_{k=1}$ containing $K$ vectors (`K=2048` in this paper)
    - The residual $r_l$ for level $l$ is used to find the nearest codebook vector $e_l$
    - The ID associated with $e_l$ is taken as the semantic ID
    - We compute $r_{l+1} := r_l - e_l$
    - Thus we end up with a sequence of semantic IDs (e.g. `(1, 4, 6, 2)`)
- A decoder $\mathcal{D}$ maps the quantized latent $\hat{z} := \sum_{l=1}^L e_l$ back to $\hat{x}$

The RQ-VAE model is trained with the following losses:
- $\L := \L_{recon} + \L_{rqvae}$
- $\L_{recon} := ||x - \hat{x}||^2_2$ aims to reconstruct the content embedding $x$
- $\L_{rqvae} := \sum_{l=1}^L \beta \cdot ||r_l - sg[e_l]||^2 + ||sg[r_l] - e_l||^2$ where $sg$ is the stop-gradient operator (which disables gradient updates to the term in the operator). This is to encourage $r_l$ and the codebook vector $e_l$ to move toward each other, so that we have a good codebook at the end.

## Stage 2: Using Semantic IDs for Ranking

Now that each item $v$ is represented by a sequence of semantic IDs $(c_1^v, ..., c_L^v)$, we can treat each ID as a token. The most intuitive thing is to treat each token as a subword and assign a unique embedding to each token (this is the `unigram` approach below). Unfortunately, this simplistic approach is not the most ideal. The experiments reveal that these semantic tokens behave more like <<characters>> in NLP, and it is important for performance to assign unique embeddings to *sequences of tokens* (i.e. create subwords out of the characters). 

There are two approaches that were experimented:
- **Create n-grams out of the IDs**. Suppose an item has semantic IDs `(4, 6, 3, 2)`. A unigram approach would look up embeddings for each ID by itself. A bigram approach would look up embeddings for `(46, 63, 32)` and so on. Consequently, the embedding table size for a quantizer with `L` levels and `K` codes in each level is something like $(L-N+1) \times K^N$. This gets prohibitively expensive for larger `N`, so the experiments stop at bigram.
- **Sentence piece based**. As with natural language, most n-grams rarely occur (or do not occur) at all. Hence the n-gram approach is wasteful. The authors found that applying the [sentence piece approach](../nlp_course/intro.md#lecture-2-word-representation-and-text-classification) (I suppose using byte pair encoding?) is an effective way to learning the subwords to assign a unique embedding. The authors train the subword model on impressed items to a desired arbitrary vocab size. <<Note:>> this makes the approach less flexible, as we need to freeze both the quantizer and the set of subwords. However their experiments show that this is generally quite resilient.

## Results

The experiments use CTR AUC as the target metric. The model is trained on a window of N days and predictions on day N + 1. Cold start performance refers to the performance on the subset of items that were newly introduced on day N + 1. Note that each user is represented as a sequence of items, including past item history and the current item. The item is represented by semantic IDs for itself. The main findings of the experiments:
- <<Sentencepiece (SPM) approach > n-gram approaches>> for overall performance. As the vocab size is scaled up for the `SPM` approach, it outperforms both unigram and bigram easily.
- <<SPM > random hashing>> with the same vocab size for overall performance. It does not compromise on the memorization capacity of the model.
- <<All Semantic ID approaches > random hashing>> in the cold start scenario (as expected)

The authors also conducted a set of experiments where items are represented by their content embedding directly. Due to the large size of the embedding, they did not include user past history for these experiments. These showed that <<content embedding approach is inferior to random hashing>>, unless the number of layers is increased significantly. This suggests that a larger model is indeed able to memorize better just using the content embeddings, but at a significant computational cost. Hence this justifies the use of semantic IDs as a more efficient way to balance between memorization and cold start performance.

## Takeaways

Semantic IDs is a promising approach to balance memorization and cold start performance. However, it presupposes that we have good embeddings for each item. It also introduces significant engineering complexity in training and freezing a residual quantizer and a fixed subword vocabulary. Although the authors conduct experiments to show that this quantizer is quite robust to data distribution shift, there would probably come a time when we need to update the quantizer and the subword vocabulary and deal with the refreshing issue.

