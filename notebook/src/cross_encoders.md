# Cross Encoders

Cross encoder is a type of model architecture used for re-ranking a relatively small set of candidates (typically `1,000` or less) with great precision. In the Question-Answering or machine reading literature, typically the task involves finding the top matching `documents` to a given `query`. A typical task is the MS MARCO dataset, which seeks to find the top documents that are relevant to a given bing query.

## Basic Setup

Typically, the base model is some kind of pre-trained BERT model, and a classification head is added on top to output a probability. Each `(query, document)` pair is concatenated with `[SEP]` token in-between to form a sentence. The sentence is fed into the classification model to output a probability. The model is trained using binary cross-entropy loss against `0,1` labels (irrelevant or relevant).

This is the setup used by <Nogeuira 2019>, possibly the first paper to propose the cross encoder. Some specifics for their setup:
- The query is truncated to max `64 tokens`, while the passage is truncated such that the concatenated sentence is max `512 tokens`. They use the `[CLS]` embedding as input to a classifier head.
- The loss for a single query is formulated as below. $s_i, s_j$ refers to the score from the classifier model, $J_{pos}$ refers to the documents that are relevant, and $J_{neg}$ refers to documents in the top `1,000` retrieved by BM25 that are not relevant. Note that this results in a very imbalanced dataset.

$$L = - \sum_{j \in J_{pos}} log(s_j) - \sum_{j \in J_{neg}} log (1 - s_j)$$

- The model is fine-tuned with a batch size of 128 sentence pairs for 100k batches.

As opposed to bi-encoders (or dual encoders), which take a dot product between the `query` embedding and the `document` embedding, we cannot pre-compute embeddings in the cross encoder setting, because the cross encoder requires a forward pass on the concatenated `(query, document)` pair. Due to the bi-directional attention on the full concatenated sentence, we need the full sentence before we can compute the score, which requires the `query` that we only see at inference time. Hence, the cross encoder is limited to reranking a small set of candidates as it requires a full forward pass on each `query, candidate_document` pair separately.

## Contrastive Loss

The vanilla binary cross entropy loss proposed above may be thought of as a <point-wise> loss, in which each document is either relevant or irrelevant in absolute terms. However, treating relevance as a <relative> concept often better reflects reality. For example, given the first page of search results for a Google query, most of the documents should be relevant to some extent, but some are more relevant than the rest (and get clicked on).

Thus <Gao 2021> proposes the Local Contrastive Estimation loss. For a given query `q`, a positive document $d_q^+$ is selected, and a few negative documents $d_q^-$ are sampled using a retriever (e.g. BM25). The contrastive loss then seeks to maximize the softmax probability of the positive document against the negative documents.

$$
L_{LCE} = \frac{1}{|Q|} \sum_{q \in Q,\ G_q} 
    -log \frac{
        exp(s_{q,\ d_q^+})
    }{
        \sum_{d_q^- \in G_q} exp(s_{q,\ d_q^-})
    }
$$

It is confirmed in multiple experiments in <Gao 2021> and <Pradeep 2022> that LCE consistently out-performs point-wise cross entropy loss. Furthermore, the performance consistently improves as the number of negative documents per query (i.e. $|G_q|$) increases. In <Gao 2021>, up to `7` negatives (i.e. batch size of `8`) were used. <Pradeep 2022> shows that increasing the batch size up to `32` continues to yield gains consistently (albeit diminishingly).






## References

- [Nogueira 2019 - Passage Re-ranking with BERT](https://ar5iv.labs.arxiv.org/html/1901.04085)
- [Gao 2021 - Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline](https://ar5iv.labs.arxiv.org/html/2101.08751)
- [Pradeep 2022 - A Bag of Tricks for Improving Cross Encoder Effectiveness](https://cs.uwaterloo.ca/~jimmylin/publications/Pradeep_etal_ECIR2022.pdf)
