# Penha 2025 - Joint-task Semantic ID

[Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478)

This paper addresses the task of constructing semantic IDs that are useful for both search and recommendations. The main empirical finding is that constructing semantic IDs in a task specific way will necessarily degrade performance for the other task.

## Methods

The baseline / single-task methods considered for constructing semantic IDs:
- <<Content-based>>. An off-the-shelf embedder is used to embed textual content of items. This is the approach used in DSI, TIGER. Specifically, `all-mpnet-base-v2` was used on the concatenated metadata of each item. The embedding is then discretized.
- <<Search-tuned IDs>>. Starting from `all-mpnet-base-v2`, the model is fine-tuned on search data with in-batch random negatives (i.e. `MultipleNegativesRankingLoss` from `sentence-transformers`). The search data comprises of (`search_query`, `relevant_item_metadata`) pairs. The fine-tuned embedding is then discretized.
- <<Recommendation-tuned IDs>>. The Efficient Neural Matrix Factorization (EMNF) method from TokenRec is used to create collaborative filtering based embeddings $v^{rec}$. 

## Multi-Task Methods

A few methods were explored for how to combine multi-task signals:

- <<Separate>>. This means that each task has its own set of item IDs. At inference time, search prompts can only output search tokens, and recommendation prompts can only output rec tokens.
- <<Fused Concat>>. Each embedding $v^{search}$ and $v^{rec}$ are individually l2-normalized and concatenated together.
- <<Fused SVD>>. Each embedding is l2-normalized but we further dimension reduce the embedding with higher dimensional space using truncated SVD. The two embeddings are then element-wise added together.
- <<Multi-task>>. A bi-encoder is trained on both supervision signals:
    - `(query, item)` pairs from search data and `(item_a, item_b)` pairs from interaction data.

## Semantic-id Learning Methods

The methods considered for learning semantic ID:
- <<RQ-kmeans>>. This is basically hierarchical k-means on the residuals at each level, implemented using FAISS residual quantizer.
- <<RQ-VAE>>. Implemented using `vector-quantize-pytorch`
- <<MiniBatchDictionaryEncoding>>. From `sklearn` library
- <<ResidualLFQ>> from `vector-quantize-pytorch`

## Data / Metrics

A search and recommendation dataset is built from `MovieLens25M`:
- `62k` movies
- `1.2M` user-item interactions (last item per user used for test)
- `20` queries per item generated using `gemini-2.0-flash`

Recall at 30 is used as evaluation metric. `google/flan-t5-base` is used as the generative model to learn to output semantic IDs given the context (be it search or user history) using supervised fine-tuning.

## Results

The results show that:
- Single-task based embeddings perform best in their own task, much better than any multi-task method. But they degrade performance on the other task very badly.
- Amongst the multi-task methods, the <<multi-task>> method of training a bi-encoder on both contexts works best. 
- Amongst semantic ID tokenisation methods, `RQ-Kmeans` was the clear winner, far outperforming all other methods. `RQ-VAE` in particular showed degenerate results especially for the search case.

> <<Question:>> Why did RQVAE perform so badly compared to the naive RQ-Kmeans? Did the authors handle the RQVAE correctly?

