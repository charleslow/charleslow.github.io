
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

# Lecture 5: Transformers

[Lecture 5 Link](https://www.youtube.com/watch?v=QkGwxtALTLU)

Transformers started as cross attention (Bahdanau 2014) and evolved to self-attention (Vaswani 2017). There are two main types of transformers:
1. <Encoder-Decoder> model, e.g. T5 or BART. Encoder-decoder models have a separate encoder and decoder module. The encoder takes the input sequence and passes it into a transformer block to return a "context" embedding. The decoder then takes in both the "context" embedding and the output sequence (e.g. the same sentence in French for translation, or the same sentence itself shifted right for self-attention) with appropriate masking and generates a probability for each position in the output sequence. Note that the encoder module has un-masked full attention over the entire input sequence, but the decoder module is masked.

    The benefit of the encoder-decoder model is that we get an embedding representation of a given input sequence which can be used for downstream tasks. But it is also less flexible because we need to have a concept of the "input sequence" vs the "output sequence", which is suitable for tasks like translation but not in text generation tasks there is no clear input/output.

2. <Decoder only> model, e.g. GPT or LLaMa. The decoder model simply takes in the input sequence and passes it through a transformer module, resulting in a probability for each position in the input sequence. Using appropriate causal masking, we can train the network to predict the next token at each position given only information about tokens at previous positions.

    The benefit of the decoder model is fewer parameters than the encoder-decoder model. It's also more suitable for text generation tasks. In the encoder decoder framework, at each time step we need to recompute the encoder representation, because the representation at earlier positions can change with the addition of a new token. In the decoder only framework, the previous cached Q, K, V values do not change due to the causal masking, so we can re-use those in decoding the next time step. We should thus expect decoder only models to be faster at decoding.

## Core concepts

<Multi-head attention>. The intuition for having multiple attention heads is that information from different parts of the sentence can be useful to disambiguate in different ways. Typically for a given word to attend to nearby words is useful for learning syntax, but attending to further words is useful for learning semantics.

Multi-head attention basically comprises of multiple attention modules, each with their own weights $W_{Q_i}, W_{K_i}, W_{V_i}$. The attention outputs are computed for each head, and then concatenated together. Since the resulting matrix will have $n_{head}$ times the size on one dimension, we pass it through a final linear transformation to get the desired dimension size (as though we only used one attention head).

In practice the attention weights across all the heads are concatenated together first before the matrix multiplication to vectorize the computation. The resulting matrix is then sliced and attention computed on each head, before concatenating together again for the final matrix multiplication.

<Positional Encoding>. There is no notion of position in the transformer. Positional encoding simply adds an embedding at each position to the word embedding to encode this information. Note that it is added right at the beginning to the raw token embeddings. 

1. <Sinusoidal Encoding> was the original proposal in the Vaswani 2017 paper. The position embedding is a fixed vector at each position $t$, where the $i^{th}$ element is $sin(\omega_k \cdot t)$ for even $i = 2k$ and $cos(\omega_k \cdot t)$ for odd $i = 2k+1$, and $\omega_k := \frac{1}{10000^{2k/d}}$. Here $k$ is an index on the dimension going from $1, ... \frac{d}{2}$ and $d$ is the dimension of the embedding. 

    The method is rather counter-intuitive but the basic idea is that we want the dot product between two embeddings to be high when the relative position is near and decay as we move away. The intuition is covered nicely in [Kazemnejad 2019](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) and basically we can think of each positional embedding as analogous to a binary encoding represented by $0, 1$ bits at each dimension. Instead of a hard $0$ or $1$, the $sine$ and $cosine$ functions provide a smoothed version so that we get a nice relative decay. 

    Note that this method does allow us to extrapolate the position embedding to longer sequences that we have not seen before in training.

2. <Learnable Embeddings>. This was proposed in Shaw 2018, which basically just allows the embedding at each position to be a learnable vector. The problem of this approach is that it becomes impossible to extrapolate to longer sequences in inference time.

3. <Rotary Positional Encodings> or <RoPE>. This was proposed in Su 2021. The fundamental idea is that we want the dot product of embeddings to result in a function of relative position.

    Specifically, we desire that for given positions $m, n$, and the respective word embeddings $x_m, x_n$, the dot product of the resulting embeddings may be expressed purely as a function of their relative distance $m-n$, and in so doing we lose notion of the absolute position entirely. 
    $$
    f_q(x_m, m) \cdot f_k(x_n, n) = g(x_m, x_n, m - n)
    $$

    The paper uses trigonometry and imaginary numbers to come up with a function that satisfies this property. The benefit of losing notion of absolute position entirely means that we can extrapolate to longer sequences that we have not seen before, and RoPE extrapolates better than sinusoidal embeddings. LLaMa uses RoPE embeddings.

<Stability>. Problem of gradient vanishing or exploding as we pass through the layers of an rnn or transformer. <Layer normalization> (Ba 2016) is the traditional way to deal with this issue. The intuition is that it normalizes the outputs of each attention layer to a consistent range, preventing too much variance in the scale of outputs.

$$
\text{LayerNorm}(X;g,b) = \frac{g}{\sigma_x} \cdot (X - \mu_x) + b
$$

Here, $X \in \R^{d \times n}$ is the output of an attention layer of embedding dimension $d$ and sequence length $n$, $\mu_x$ and $\sigma_x$ are the element-wise mean and standard deviation respectively across the time positions, and $g$ and $b$ are learnable vectors of dimension $d$. Hence, if $X$ has very large or small values, the normalization process standardizes the range of values. The parameters $g$ and $b$ allow the model flexibility to shift the values to a different part of the space.  

A simplification of layer norm is <RMSNorm> (Zhang and Sennrich 2019). It removes the mean and bias terms but does not hurt performance empirically. It is used in Llama. The only learnable parameters per layer is $g$.
$$
\begin{align*}
    RMS(X) = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}\\
    RMSNorm(X) = \frac{X}{RMS(X)} \cdot g
\end{align*}
$$

<Residual Connections>. Add an additive connection between input (to an attention layer) and the output.
$$
    \text{Residual}(X, f) = f(X) + X
$$

Where $f$ is the attention layer function. It prevents vanishing gradients (since we get $X$) and also allows $f$ to focus on learning the difference from the input. In the self-attention context, having the residual connection also prevents the need for tokens to attend to themselves, since that is provided by $X$ itself.

<Post vs Pre Layer Norm> (Xiong 2020). This paper found that applying LayerNorm after the attention layer (and the residual connection) broke the residual connection, because we have $\text{LayerNorm}(f(X) + X)$, which means that we are no longer guaranteed to get the input $X$. This led to some instability in transformer training. Instead, they found it is better to apply $f(\text{LayerNorm}(X)) + X$, which led to more stable training, since the residual connection is preserved.

<Activation functions>.
- Vaswani used $ReLU(x) = max(0, x)$
- LLaMA uses Swish or SiLU (Hendricks and Gimpel 2016), which is
$$
    Swish(x; \beta) = x \cdot \sigma(\beta \cdot x)
$$
    Looks a lot like ReLU but avoids the zero gradient problem.

Transformers are powerful but fickle - Vaswani 2017 used Adam with learning rate increase and decrease (warm up). This is no longer that necessary after the pre-layer norm (Xiong 2020).

<AdamW> (Loshchilov and Hutter 2017) is now more popular. It applies weight decay for regularization to Adam.

In summary, some comparisons. The Mamba paper found that the LLaMA architecture is 10x more efficient in terms of scaling law compared to the original Vaswani.

| | Vaswani | LLaMA |
|: --- :|: ---- :|: ---- :|
| Norm Position | Post      | Pre     |
| Norm Type     | LayerNorm | RMSNorm |
| Non-linearity | ReLU      | SiLU    |
| Position-Encoding | Sinusoidal | RoPE |

# Lecture 6: Decoding Strategies

What is an LLM? It is a model that defines a conditional probability distribution of a sequence given some input sequence $X$.

$$
    P(Y|X) = \prod_{j=1}^J P(y_j | X, y_1, ..., y_{j-1})
$$

The nice thing about the conditional distribution is that we get some notion of the model's confidence about the next token to generate. The problem with the conditional distribution is <hallucination>, since models generally assign some small but non-zero probability to  incorrect tokens, even if all the pre-training data is factual. See Kalai and Kempala 2023.

<Ancestral Sampling> is to sample the next token based on the indicated confidence by the model. The nice thing is that the resultant generations follow exactly the distribution of the model.

$$
    y_j \sim P(y_j | X, y_1, ..., y_{j-1})
$$

The problem with ancestral sampling is the <long tail> problem. Most language models have around 30k tokens, and the probabilities from the long tail adds up, so that there's a somewhat good chance of sampling something really unlikely. 

The obvious solution to this problem is to ignore the long tail and only sampling from the top-k most probably tokens. This is <top-k sampling>. This results in only tokens that the model is somewhat confident in. 

Alternatively, we could only sampling from the top-p probability mass. This is called <top-p or nucleus sampling>. This is to account for the case where top-k sampling is not so desirable because if most of the probability mass is only on say 3 tokens, we may only want to sample from them. 

Another alternative is <epsilon sampling>, where we only sample tokens with some minimum probability. This ensures that we only sample from tokens where the model is somewhat confident.

Another strategy is to modify the "peakiness" of the data by controlling the <distribution temperature>. This is done by modifying the scaling factor for the final softmax layer. This allows the user to put more weights on the top results (`temperature < 1.0`) for factual answers, or spread the weights out more by increasing the temperature for say story generaion.

<Contrastive Decoding> is a newer idea - the idea is that we use a smaller model to improve the performance of a larger model. Instead of just decoding from the larger model's distribution, we contrastively decode by choosing outputs where the expert model thinks are more likely than the smaller model. The intuition is that both the big and small model are degenerate in similar ways (e.g. keep repeating itself), but the expert has knowledge which the smaller model does not have. Hence taking the probability difference helps to eliminate the degenerate cases. 



