# Reimers 2019 - Sentence-BERT

[Link to paper](https://ar5iv.labs.arxiv.org/html/1908.10084).

Sentence-BERT (or SBERT) was one of the first papers to suggest a way to fine-tune BERT models to generate useful embeddings that can be used for search / retrieval. 

Prior to SBERT, BERT models were mainly used for sentence pair regression tasks by passing two sentences into the transformer network and adding a classification head on top to produce a float value. We can call this the <<cross-encoder>> approach. In other words, researchers only cared about the final prediction and did not make use of the embeddings, or the final vector representation of the inputs. This approach is suitable for reranking a small number of documents but not for nearest neighbour search in a corpus with millions of documents.

Naively, one can element-wise average the BERT embeddings at the final layer to produce a vector representation of the text. This vector representation can then be used for nearest neighbour search or clustering. However, because BERT was not explicitly trained for this objective, it results in rather bad sentence embeddings, often worse than GloVe embeddings.

## Method

The SBERT paper presents 3 different training objectives, all of which perform well on embedding similarity tasks. The choice of objective depends on the dataset:

1. <<Classification objective>>. This is for tasks where the objective is to predict a label given two sentences A, B. We pass each sentence into the BERT network and a pooling layer to get two vector representations, $u$ and $v$. The pooling layer can be (i) take the `[CLS]` token embedding, (ii) take the element-wise mean or (iii) take the element-wise max. We then create a concatenated vector $(u,\ v,\ |u-v|)$ which is fed into a softmax classifer. The network is trained using cross-entropy loss.

    Note that this siamese approach (where each sentence is passed into the same network) differs a little from the typical cross-encoder approach, where the sentences are concatenated as a <<string>> with the `[SEP]` token before passed into the network. The latter approach is presumably more powerful because the attention mechanism can attend to all pairwise relationships

2. <<Regression objective>>. This is for tasks where the objective is to predict a float given two sentences A, B. Given the vectors $u$ and $v$, the cosine similarity is simply taken to generate a float between $-1$ and $1$. The cosine similarity is then compared with the actual float value using mean-squared error to generate a loss.

3. <<Triplet objective>>. This is for tasks where each data point is a triplet (anchor sentence $a$, positive sentence $p$, negative sentence $n$). We then minimize the loss function, where $m$ is the margin:
$$max(||s_a - s_p|| - ||s_a - s_n|| + m, 0)$$

## Ablation

1. <<Pooling strategy>>. Using `[CLS]` or mean seems to be largely similar. The authors saw some degradation using max for the regression objective.
2. <<Concatenation>>. For the classification objective, the concatenation strategy makes some difference. In particular, using $(u, v)$ yields $\rho=0.66$ but $(u, v, |u-v|)$ yields $\rho=0.81$. Thus the element-wise difference is important in yielding useful embeddings, probably because it can be used to push similar sentences together and dissimilar sentences apart. The authors also found that adding element-wise multiplication $u * v$ does not help.

## Takeaway

It is interesting that the classification objective, which is close to a cross-encoder framework, is also able to learn useful embeddings by adding the difference operation $|u-v|$. This suggests that we can train a cross encoder and simultaneously get useful embeddings for nearest neighbour retrieval at the same time.
