
# Lecture 1: Introduction

[Lecture 1 Link](https://www.youtube.com/watch?v=6NeTO61qc4M)

A general framework for NLP systems: create a function to map an input X into an output Y, where X and/or Y involved language. Tasks include:
- Language Modelling
- Translation
- Text Classification
- Langugage Analysis
- Image Captioning

Challenges of NLP
- Low frequency words
- Conjugation (e.g. adjectives modifying a word)
- Negation
- Metaphor, analogy
- Other languages

Topics:
1. Language Modelling Fundamentals
2. Training and Inference Methods
3. Experimental Design and Evaluation
4. Advanced Training and Architectures
5. NLP Applications
6. Linguistics and Multi-Linguality

# Lecture 2: Word Representation and Text Classification

<Subword models> aim to address issue of low-frequency words. How many words are there in the English language? This is a bit of a trick question because do we consider `company` or `companies` different words? By using subwords we can greatly reduce the vocabulary size to around `60k` which is used in modern tokenizers. 

Another way to address this is to use character-level models. The main issue of character-level models is that our sequences become very long, so for a fixed sequence length we cover a lot less text in the same batch size. Using subwords saves compute and memory.

<Byte Pair Encoding> (2015) is a very simple method to create subwords. It simply incrementally combines the most frequent token pairs together, starting at the character level. e.g. starting with the sentence `newest, widest`, we first combine `es` into a token, then `est`, and so on.

Another way to do this is to use <Unigram Models> (e.g. Kudo 2018). First we use a unigram LM that generates all words in the sequence independently. We then pick a vocabulary that maximizes the log likelihood of the corpus given a fixed vocabulary size. The optimization process is using the EM algorithm.

`Sentencepiece` is a highly optimized library to train both these types of subword models.

Subword nuances:
- Subword models are hard to use multilingually because they will over-segment less common languages naively
- Subword segments are sometimes arbitrary (e.g. is it `es t` or `e st`)?

Pytorch vs Tensorflow / JAX
- Pytorch is more widely used
- Pytorch favour dynamic execution vs compile + execute

Training transformers typically uses a learning rate schedule. Start low, increase in the middle and decrease toward the end. Starting with a high learning rate will lead to weird models because transformers are sensitive. 

# Lecture 3: Language Modelling

[Lecture 3 Link](https://www.youtube.com/watch?v=69EAJOwV3Es)

Generative vs Discriminative models:
- Discriminative model: a model that calculates the probability of a latent trait given the data, i.e. $P(Y|X)$, where $Y$ is say the sentiment and $X$ is the text
- Generative model: a model that calculates the probability of the data itself i.e. $P(X)$

Generative models for text are called language models. We can use LMs to generate sentences. We can also use them to score sentences (e.g. when a sentence is a concatenation of a question and answer pair). Can also use LMs to correct sentences.

Auto-regressive language models compute $P(X) = \prod_{i=1}^I P(x_i | x_1, ..., x_{i-1})$. Question: why do we do it auto-regressively instead of computing the entire sequence $x$ at once? The problem is computational - predicting the next token has a space of $|\mathcal{V}| \sim 60,000$, but predicting the entire sequence is in the order of $|\mathcal{V}|^N$ where $N$ is the length of the sequence, which is currently untractable. That being said, if we can model the entire sequence, it will probably be a lot more efficient than the auto-regressive way.

The simplest language model is <count-based unigram model>. By making an indepenence assumption $P(x_i | x_1, ..., x_{i-1}) \sim P(x_i)$, we just ignore all the previous words and predict the probability of a word occurring.

The maximum likelihood estimation for the probability of a given token $x_i$ will simply be:

$$
    P_{MLE}(x_i) = \frac{c_{train}(x_i)}{\sum_{\widetilde{x}}(\widetilde{x})}
$$

Detail: parametrizing in log space. Multiplication of probabilities are re-expressed as additions in log space, because otherwise, if we multiply 100 probabilities together for a sequence of 100 tokens, we will easily underflow the numeric precision.

$$
    P(X) = \prod_{x=1}^{N} P(x_i)\\
    log P(X) = \sum_{x=1}^{N} log P(x_i)
$$

Correspondingly, we can define the parameters $\theta_{x_i} = log P(x_i)$.

Moving on to <higher order n-gram models>, the idea is to limit the context length to one-word before the token we are predicting, and then count:

$$
    P_{ML}(x_i | X_{i-n+1}, ..., x_{i-1}) := \frac{c(x_{i-n+1}, ..., x_i)}{c(x_{i-n+1}, ..., x_{i-1})}
$$

e.g. P(example | this is an) = c(this is an example) / c(this is an).

Due to sparsity of data, we need to add smoothing to deal with zero counts, i.e. instead of just using tri-gram, we smooth tri-gram and bi-gram probabilities together:

$$
    P(x_i | x_{i-n+1}, ..., x_{i-1}) = \lambda P_{ML}(x_i | x_{i-n+1}, ..., x_{i-1}) + (1-\lambda)P(x_i | x_{i-n+2}, ..., x_{i-1})
$$

e.g. $\text{P(example | this is an)} = \lambda P_{ML} \text{(example | this is an)} + (1-\lambda) P_{ML} \text{(example | is an)}$.

More sophisticated smoothing techniques are studied in [Goodman 1998: An Empirical Study of Smoothing Techniques for Language Modelling](https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf).

Problems:
1. Cannot share strength amongst similar words, e.g. `car` and `bicycle`
2. Cannot condition on context with intervening words, e.g. `Dr Jane Smith` vs `Dr Gertrude Smith`
3. Cannot handle long-distance dependencies, e.g.`tennis` and `racquet` in `for tennis class he wanted to buy his own racquet`

The standard toolkit for n-gram models is [kenlm](https://github.com/kpu/kenlm) which is extremely fast and scalable, written in c++.

## Evaluating Language Models

1. Log Likelihood:

$$
    LL(X_{test}) = \sum_{X \in X_{test}} log P(X)
$$

2. Per-word Log Likelihood:

$$
    WLL(X_{test}) = \frac{1}{\sum_{X \in X_{test}} |X|} \sum_{X \in X_{test}} log P(X)
$$

3. Per-word Cross Entropy:

$$
    H(X_{test}) = -\frac{1}{\sum_{X \in X_{test}} |X|} \sum_{X \in X_{test}}  log_2 P(X)
$$

Aside: Any probabilistic distribution can also be used to compress data. The entropy measure is closely related to the number of bits needed to store the data based on our language model.

4. Perplexity. Lower is better. Perplexity is the number of times we need to sample from the probability distribution until we get the answer right.

$$
    PPL(X_{test}) = 2^{H(X_{test})} = e^{-WLL(X_{test})}
$$

## Other Desiderata of LMs

<Calibration> Guo 2017. Formally, we want the model probability of the answer matching the actual probability of getting it right. Typically we measure calibration by bucketing the model output probabilities and calculating <expected calibration error>:

$$
    ECE = \sum_{m=1}^M \frac{|B_m|}{n} |acc(B_m) - confidence(B_m)|
$$

where $m$ represents a sub-segment of the data which corresponds to a confidence interval from the model.

How do we calculate answer probabilities? e.g. `the university is CMU` and `the university is Carnegie Mellon University` should both be acceptable. 
- One way is to use paraphrases to substitute phrases `Jiang 2021`.
- One way is to just ask the model to generate a confidence score, see [Tian 2023 - Just ask for calibration](https://aclanthology.org/2023.emnlp-main.330.pdf)

Another desirable characteristic is <efficiency>. Some metrics are:
- Memory usage (load model only, peak memory usage)
- Latency (to first token, to last token)
- Throughput

Some efficiency tips:
- On modern hardware doing 10 operations of size 1 is much slower than doing 1 operation of size 10
- CPUs are like motorcycles and GPUs are like airplanes
- Try to avoid memory moves between CPU and GPU, and if we need to move memory, do so as early as possible (as GPU operations are asynchronous).

# Lecture 4: Sequence Modelling

[Lecture 4 Link](https://www.youtube.com/watch?v=x3U2zVhrgJ8)

NLP is full of sequential data, especially those containing long range dependencies. References can also be complicated, e.g. `the trophy would not fit in the suitcase because it was too big`. What does `it` refer to? These are called <winograd schemas>.

Types of tasks:
- Binary classification
- Multi-class classification
- Structured Prediction. e.g. predicting the parts-of-speech tag for each word in the sentence

Sequence labelling is an important category of tasks in NLP:
- e.g. Parts of speech tagging
- Lemmatization
- Morphological tagging, e.g. `PronType=prs`

Span labelling:
- Named entity resolution
- Syntactic Chunking
- Entity Linking
- Semantic role labelling

We can treat span labelling as a sequence labelling task by using `beginning`, `in` and `out` tags.

Three major types of sequence modelling:
1. Recurrence
2. Convolutional 
3. Attentional

Recurrent neural networks essentially unroll a computational graph through "time" (where time is the position in the sequence). This results in a <vanishing gradient> problem, where the gradient on the last token is largest, and the gradient becomes smaller as we move backwards towards the first token. This problem is not just present in RNNs, but in general for computation graphs, if there is important information, adding direct connections from the important nodes to the loss is one way to improve performance. This is the motivation for residual or skip-connections.

<LSTM> is one way of solving this problem - the basic idea is to make additive connections between time steps, which does not result in vanishing. The idea is to have different "gates" that control the information flow, and use additive connections. Additive connections solves the vanishing gradient problem because it does not modify the gradient (?).

Attention score functions:
- Original paper in Bahdanau 2015 used a multi-layer perceptron for the attention function, which is very expressive but has extra parameters
$$
    a(q, k) = w_2^T tanh(W_1 [q;k])
$$

- Bilinear in Luong 2015:
$$
    a(q, k) = q^T W k
$$

- Dot product:
$$
    a(q, k) = q^T k
$$

- Scaled Dot product (Vaswani 2017):
$$
    a(q, k) = \frac{q^T k}{\sqrt{|k|}}
$$

Note that the attention mechanism in Vaswani 2017 is essentially a bilinear function + scaling, because the weight matrices are involved in obtaining the query and key vectors.

Comparing RNN, convolution and attention, for a token at position `200` to attend to token at position `1`:
- RNN will take 199 steps through the computation graph
- Convolution will take 20 steps if the convolution window is 10
- Attention will take 1 step

This is an advantage of the attention representation of text. Another advantage of attention is the computation speed. For a given text of `N` tokens, in order to produce `N-1` token predictions for the next word at each step:
- RNN will have to sequentially run `N-1` steps
- Attention will do it in 1 step by masking the attention matrix suitably. This makes it much faster to train attention models.

<Truncated Backpropagation> is one technique to reduce computation. e.g. in the context of RNN, we might forward propagate through positions `1-10` and compute the gradients. For the next positions `11-20`, the hidden state from position `10` is used for forward propagation, but we do not backpropagate the gradients back to positions `1-10`.