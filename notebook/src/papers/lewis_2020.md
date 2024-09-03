# Lewis 2020 - Retrieval Augmented Generation

[Lewis 2020 - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://ar5iv.labs.arxiv.org/html/2005.11401)

This paper proposes a way to fine-tune a parametric seq2seq transformers (e.g. GPT) with a non-parametric memory through dense retrieval. The main idea is to extend <<parametric memory>> (i.e. the "knowledge" that is stored within the LLM floating point parameters) of the seq2seq model by coupling it with retrieval of documents from a vector database (dubbed <<non-parametric>> memory, using Wikipedia articles in the original paper). We can update the non-parametric database's knowledge as the world changes.

The paper argues that this setup is ideal for *knowledge-intensive* tasks such as open question answering, fact verification etc., where it is impossible to store all knowledge in <<parametric memory>>.

## Setup

Given an input sequence of tokens $x$, we wish to retrieve contextual documents $z$ and use them as additional context when generating target sequence $y$. We have two models:
- <<Retriever>> $p_{\eta}(z | x)$ which returns top-k truncated probability distributions over text passages. Truncated probably means that the probability is normalized such that the top-k probabilities sum up to `1` for each retrieval.
- <<Generator>> $p_{\theta}(y_i | x, z, y_{1:i-1})$ parametrized by $\theta$ which generates tokens in an auto-regressive manner.

The goal is to train both models end-to-end on a fine-tuning corpus of input sequence / output sequence pairs. The loss objective is to minimize the negative log-likelihood $-log \ p(y_j | x_j)$ .

## Training

The paper proposes two different models to achieve the end-to-end prediction.
1. <<RAG-Sequence>>. In this setting, we retrieve `k` documents and keep using them to generate the entire target sequence. 

$$
\begin{align*}
    p_{RAG-Sequence}(y|x)
        &\sim 
        \sum_{z \ \in \ topk \{ p_{\eta}(\cdot | x) \}}
            p_{\eta}(z|x) p_{\theta}(y|x, z) \\
        &= 
        \sum_{z \ \in \ topk \{ p_{\eta}(\cdot | x) \}}
            p_{\eta}(z|x) \prod_{i}^N p_{\theta}(y_i |x, z, y_{1:i-1})
\end{align*}        
$$

- Note that we are marginalizing (or taking a weighted combination) over the truncated distribution of $z$, implying that we *trust* each document according to its retrieval probability in the final probability for generating each token.
- The expression $p_{\theta}(y_i | x, z, y_{1:i-1})$ just means that we generate the target sequence with an input sequence that is a concatenation of $x$, $z$ and $y_{1:i-1}$.
- The retrieval is done using a BERT encoder using Maximum Inner Product Search (MIPS). To avoid re-indexing, the document vectors are held constant whilst the query encoder is trained in the above end-to-end fashion.
- There is no explicit supervision on what documents are to be retrieved. Intuitively, if a document is useful for generating the correct tokens, the loss objective would encourage $p_{\eta}(z|x)$ to be larger, thus encouraging the retriever to retrieve more relevant documents.
- Another interesting way to think about this setup: suppose the generator just returns the retrieved document (token for token) and `k=1`, and the input / output pairs are `anchor-positive` pairs in a standard retrieval setting. Then we can see that this matches the standard retrieval training objectives but *without negative sampling*. Thus it seems that the token prediction task is sufficient to generate negative signals for non-useful documents such that explicit negatives are not needed.

2. <<RAG-token>>. In contrast, RAG-token retrieves `k` documents at each time step, allowing us to sample new documents according to the needs at each decoding step.

$$
\begin{align*}
    p_{RAG-Token}(y|x)
        \sim
        \prod_{i}^N 
        \sum_{z_i \ \in \ topk \{ p_{\eta}(\cdot | x, y_{1:i-1}) \}}
            p_{\eta}(z|x) p_{\theta}(y_i |x, z_i, y_{1:i-1})
\end{align*}        
$$

Note that the retrieved context is now $z_i$ which varies at each time step. The change in retrieval at each step seems to add complexity during training.

## Ablation

- <<Increasing number of retrieved documents>>. `5` or `10` documents are used for retrieval. Ablation shows that performance increases monotonically (albeit diminishingly) for `RAG-sequence` with increasing number of retrieved documents.
- <<Learned Retrieval is useful>>. The authors try freezing the retriever and compare it against the setting of allowing the retriever to learn. They find that generally learned retrieval improves results significantly.
- <<RAG generates more diverse outputs>>. They measure the ratio of `distinct tri-grams / total tri-grams` and find that RAG-decoding compared to normal decoding is more diverse.