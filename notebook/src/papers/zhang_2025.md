# Zhang 2025 - Qwen3 Embedding

[Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176)

This short paper introduces the Qwen3 Embedding and reranker series, which are the strongest open source models currently for such tasks. The Qwen3 foundation decoder LLM serves as the backbone for fine-tuning and also is used to generate high quality synthetic training data. Note that Qwen3 is multi-lingual and are publicly available under the Apache 2.0 license, which means it can be used commercially.

## Characteristics

The embedding and reranking models come in 3 sizes: `0.6B`, `4B` and `8B`. 
- `0.6B`: 28 layers, embedding dimension of `1024` for the embedder
- `4B`: 36 layers, embedding dimension of `2560` for the embedder
- `8B`: 36 layers, embedding dimension of `4096` for the embedder

Since the `4B` and `8B` have same number of layers, presumably the `8B` model has larger hidden sizes.

All the models have `32K` sequence length limit, and are instruction aware, meaning that we can adjust the instruction at the start of the prompt to adjust the behaviour of the embedder or reranker. For the embedding models, there is also MRL support (Matryoshka Representation Learning), meaning that we can use custom dimensions from the embeddings.

## Embedder

The text embeddings are obtained by appending an `[EOS]` token at the end of every input sequence. The final embedding is derived from the hidden state of the last layer corresponding to this `[EOS]` token.

Input format for queries or Documents is as follows:
```
{Instruction} {Query or Document}<|endoftext|>
```

The contrastive loss based on InfoNCE is used for training the embedder. Specifically, given a batch of $N$ training instances, the loss is defined as:
$$
    \L_{embedding} = -\frac{1}{N} \sum_i^N \log \frac{
        e^{s(q_i, d_i^+) / \tau}
    }{
        Z_i
    }
$$ 

$s$ is cosine similarity function, $\tau$ is temperature and $Z_i$ is the normalization factor which includes the positive pair `+` various negative pairs:
$$
    Z_i = e^{s(q_i, d_i^+) / \tau} 
    + \sum_k^K m_{ik} e^{s(q_i, d_{i,k}^-) / \tau}
    + \sum_{j \neq i}^N m_{ij} e^{s(q_i, q_j) / \tau}
    + \sum_{j \neq i} m_{ij} e^{s(d_i^+, d_j) / \tau}
$$

Comment on the above normalization factor:
- The second term is the similarity between each anchor query and $K$ hard negatives $d_{i,k}^-$ per query. Note that as it is written, only the hard negatives in the same row are used as negatives for each anchor query, but in theory we could use all negatives in the mini-batch.
- The third term is the similarity between pairs of queries. The assumption is that randomly selected queries should be unrelated to each other.
- The last term is the similarity between the positive document in each row (i.e. $d_i^+$) and all other documents $d_j$ (including hard negatives).

The $m_{ij}$ and $m_{ik}$ are mask factors designed to reduce impact of false negatives in the normalization factor $Z_i$. Specifically, given an anchor query or document $i$ and a potential negative query or document $j$:
$$
m_{ij} = 
\begin{cases}
    0 & \text{if } s_{ij} > s(q_i, d_i^+) + 0.1 \text{ or } d_j == d_i^+, \\
    1 & \text{otherwise}
\end{cases}
$$

This means that for each row, we use the similarity score between the query $q_i$ and $d_i^+$ as a dynamic threshold to filter out false negatives. For any term in $Z_i$ which has too high similarity exceeding this threshold (plus a small margin), we reject the false negative and mask it out. Note that this approach is reminiscent of triplet loss semi-hard masking or the GISTEmbed loss.

## Reranker

The reranker is simpler, and training remains in the text paradigm. Specifically, the authors use the LLM chat template to incorporate instruction and frame the reranking task as a `yes` or `no` question:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and 
the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>

<|im_start|>user
<Instruct>: {Instruction}
<Query>: {Query}
<Document>: {Document}<|im_end|>

<|im_start|>assistant
<think>\n\n</think>\n\n
```

Instead of fitting a classifier head, no change is made to the architecture. The reranking score is computed as the likelihood ratio of the next token being `yes` or `no`:
$$
    score(q,d) = \frac{
        e^{P(yes | I, q, d)}
    }{
        e^{P(yes | I, q, d)} +
        e^{P(no | I, q, d)}
    }
$$

The task then reduces to a supervised fine-tuning task, where the label is either `yes` for positives or `no` for negatives. The loss is simply the log probability of the correct label for each row (`yes` or `no`).
$$
    \L_{reranking} = - \log p(l | q, d)
$$

## Multi-stage Training

The multi-stage training has emerged as a common practice for training text embedding models. The 3 stages used are as follows:
- <<Stage 1: Large scale synthetic data>>. The Qwen 32B model was used to synthesize training pairs of data across many tasks, such as retrieval, classification, semantic textual similarity. 
    - To create diversity, a document is taken from the Qwen3 training corpus, and top 5 similar documents are retrieved
    - Qwen3 is presented with these documents and a user persona to generate a potential query
    - Qwen3 is also instructed to vary the query type, length, difficulty and language for each query
    - 150 million query - document pairs are generated this way
- <<Stage 2: High quality synthetic data>>
    - The 150 million pairs in Stage 1 are filtered down to 12 million high quality pairs
    - Specifically, only query - document pairs with cosine similarity greater than 0.7 are selected
- <<Stage 3: Model merging>>
    - Model merging based on Spherical Linear Interpolation is used, which merges multiple model checkpoints saved during the fine tuning process.

Note that all 3 stages were used for the embedder, but stage 1 was omitted for the reranker as it did not help. The ablation studies show that all 3 stages are crucial for final performance of the `0.6B` embedding model.

