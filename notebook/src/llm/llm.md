# Large Language Models

LLMs are generally used in an <auto-regressive> way, where the user supplies a prompt and the LLM returns a generated response. This framework makes it amenable to a wide range of tasks.

HuggingFace has a great [blog post](https://huggingface.co/blog/optimize-llm) explaining how we can run LLMs on humble hardware. Typically, LLMs have billions of parameters. The following rules of thumb helps us know how much memory we need to load the LLM into memory. For a model with `X billion` parameters:
- Loading in `float32` requires `4X GB` of VRAM
- Loading in `bfloat16` requires `2X GB` of VRAM
- Loading in `int8` requires `X GB` of VRAM

Hence we see that we can load a `~7 billion` parameters model with around `14GB` of VRAM if loaded in `bfloat16`, which makes it feasible to run on GPUs like Tesla T4 with `16GB` of VRAM. This can be done when loading the model with `from_pretrained(..., torch_dtype=torch.bfloat16)`. Most models are trained in `bfloat16` anyway, so it makes sense to load them at that precision.

Current popular open source LLM of that size includes `mosaicml/mpt-7b`, which can be easily downloaded and used using huggingface.

## Quantization

It turns out that we can lower the precision of models even further than 16 bits if we use a quantization method (see e.g. [Dettmers 2022](https://arxiv.org/abs/2208.07339), this paper is the basis for the package `bitsandbytes` used for quantization). The general idea is akin to encoding - we encode each number from a higher precision into a "codeword" in the lower precision (i.e. *quantization*). Numbers that are close to one another in the higher precision may get mapped to the same "codeword". When we want to use the encoded value, we look up the value in the higher precision that it maps to (i.e. *de-quantization*).

When applying this to quantizing a neural network, the steps involved are:
1. Quantize all model weights to target precision (e.g. `int8`)
2. Pass the input vector at `bfloat16`
3. At each layer, dequantize the weights and perform matmul in `bfloat16`
4. Quantize the weights again for storage

Hence while quantization lowers the memory footprint of the model, it may increase inference time. To use quantization, we need to do the following (also make sure `bitsandbytes` is pip installed). We can also pass `load_in_4bit=True` for 4bit quantization. More info on quantization usage is available at [HuggingFace](https://huggingface.co/docs/transformers/main_classes/quantization#general-usage). An important note is that a GPU is required for quantization, at least in the `bitsandbytes` package.

```python
model = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)
```

## Flash Attention

The self-attention mechanism ([Dao 2022](https://arxiv.org/abs/2205.14135)) is at the heart of the transformer performance but is also a major bottleneck in terms of memory and computational cost. One of the most successful optimizations for the attention mechanism is the Flash Attention paper. 

Suppose we have an input sequence of embeddings $X = (x_1, ..., x_N)$ where $x_i \in \mathbb{R}^k$, such that $X \in \mathbb{R}^{k \times N}$. The transformer stores parameters $W_q, W_k, W_v \in \R^{k \times d}$, such that $Q = X^T \cdot W_q, \quad K = X^T \cdot W_k, \quad V = X^T \cdot W_v$ such that $Q, K, V \in \R^{N \times d}$. The self-attention matrix $S = Q K^T \in \R^{N \times N}$ is then computed to represent the pairwise interaction between tokens at position $i$ ($i^{th}$ row) and position $j$ ($j^{th}$ column). The row-wise softmax is taken $P = softmax(S)$ to convert these into probabilities and finally the output is $O = P \cdot V$.

Typically, $N$ is much larger than the hidden dimensions $k, d$, as $N$ can be `2,048` or larger. Hence the $QK^T \in \R^{N \times N}$ matrix is the bottleneck for memory and computation. The flash attention proposes to do this computation in a block-wise manner to reduce the memory usage. Furthermore, the algorithm also speeds up the computation compared to naive attention because the block-wise implementation minimizes the number of read-write operations between the faster SRAM and slower HBM. 

More details can be found in the notebook at [Dao 2022 - FlashAttention](./papers/dao_2022.md). We can utilize flash attention like so:

```python
%pip install optimum

model.to_bettertransformer()
```

Note that this is only supported for models that have implemented flash attention, e.g. `gpt-neox`, `bloom` etc.

Flash Attention is now support natively within Pytorch as `torch.nn.functional.scaled_dot_product_attention` (see [blog](https://pytorch.org/blog/out-of-the-box-acceleration/)). The usage is like below. We need `transformers>=4.36` and `torch>=2.1.1` to use it.

```python
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16)
model = BetterTransformer.transform(model, keep_original_model=False)
```

## Position Representation 

Recent innovations in position encoding has led to accuracy improvements for long input sequences. The initial attention papers used _absolute_ position embeddings. Given an input sequence of embeddings $X = (x_1, ..., x_N) \in \R^{d \times N}$, absolute position embeddings $p_1, ..., p_N \in \R^d$ are generated by the model. These position embeddings are added to the input sequence $\hat{X} = (x_i + p_1, ..., x_N + p_N)$, thereby allowing to model to use these position cues.

It turns out that fixed positional embeddings are not ideal because they require the model to learn a fixed, unique representation of each position $1, ..., N$. This does not represent language well, because the word in position i in one sentence does not necessarily serve the same purpose as a word in the same position in another sentence. Rather, it is the relative distance between words that we want to encode in the model. Furthermore, training absolute position embeddings makes it difficult for our model to generalize to texts with longer sequences than what it was trained with.

Recent papers advocate for relative positional embeddings, with the following differences:
- Relative position encoding rather than absolute position encoding
- The encoding of relative position is done most naturally within the $QK^T$ self-attention matrix, since that is where the relative degree of interaction between tokens at different positions is encoded
- The encoding should be such that tokens further apart have a lower value in the self-attention matrix and tokens closer together have a higher value

Rotational Position Embeddings (`RoPE`) ([Su 2021](https://arxiv.org/pdf/2104.09864.pdf)) proposes rotating the query and key vectors by an angle proportional to the relative distance between the two positions. Specifically $\hat{q}_i^T \hat{k}_j = q_i^T R_{\theta,i-j} k^j$, where $R_{\theta,i-j}$ is a rotational matrix that performs the rotation.

Attention with Linear Biases (`ALiBi`) ([Press 2022](https://arxiv.org/pdf/2108.12409.pdf)) proposes an even simpler method. It simply subtracts $|i-j|/m$ from row $i$ and column $j$ of the self-attention matrix, where $m$ is a fixed scalar (specific to each attention head). Intuitively, it penalizes the attention proportional to the distance between the tokens. The study shows that this method outperforms RoPE as we extrapolate to longer sequences, and is conceptually simpler.

## Key-Value Cache

Most LLMs work in an auto-regressive manner, i.e. we provide an input sequence, generate the next token with the LLM, then append this token to the input sequence for the next iteration. Most LLMs are also trained with the causal language modelling objective and mask the upper triangle of the self-attention matrix, so that each query token $q_i$ can only interact with key token $k_j$ and value token $v_j$ if $j \geq i$. This setup encourages us to cache results from previous time steps, since a lot of computation is repeated.

The following is based on how I imagine this to work, after reading [Cameron R. Wolfe's LinkedIn post](https://www.linkedin.com/posts/cameron-r-wolfe-ph-d-04744a238_friday-ai-fundamentals-the-kv-cache-using-activity-7095825756928311296-7xXD). During training, we compute the projections $Q, K, V \in \R^{N \times d}$, where $N$ is the maximum sequence length and $d$ is the hidden dimension. The final output $O = softmax(QK^T) V \in \R^{N \times d}$ actually provides a $d$-dimension representation of the model's prediction at each of the $N$ positions. 

For next token generation, we can add a projection head, say $W_p \in \R^{d \times p}$, where $p$ represents the size of the vocabulary, such that $A = O W_p \in \R^{N \times p}$ can represent the activations *at each of the $N$ positions* for the next token. Specifically, $A_{[0,\ :]}$ represents the predictions of position $1$ given input tokens $[0]$, $A_{[1,\ :]}$ represents the predictions of position $2$ given input tokens $[0,1]$, and so on. These activations will then be fed into some cross-entropy loss such that activations at the correct token for each position gets rewarded. This allows us to do efficient training, since we simultaneously provide losses for the prediction at each of the $N$ positions to the model for backpropagation.

However, when we are doing inference generation, we only need to predict for the final position of the input sequence (suppose it is position $c$), i.e. we are only interested in $A_{[c,\ :]}$ and $O_{[c,\ :]}$. Hence for starters, we only need $q_c := Q_{[c,\ :]}$ instead of the entire $Q$ matrix, since only that row comes into play. However, we still need the entire $K$ and $V$ matrices, since we want $q_c$ to interact with all tokens in the input sequence. This is where the KV cache comes in - we cache the existing $K$ and $V$ matrices, so that we only need to project the final token of the input sequence $x_c^T W_k \in \R^{1 \times d}$ and $x_c^T W_v \in \R^{1 \times d}$ at each step and append it to the existing cached $K$ and $V$. We can then compute $O_c = softmax(q_c K^T) V \in \R^{1 \times d}$.

As one can imagine, this saves a lot of computation, but also increases memory costs. [Kwon 2023 - PagedAttention](https://arxiv.org/pdf/2309.06180.pdf) shows that serving a `13B` model on NVIDIA A100 with 40GB of memory:
- $65\%$ of memory is model parameters
- $>30\%$ is the KV cache
- A small amount of memory is used ephemerally for activation

The usage of KV cache is like so:
```python
model = AutoModelForCausalLM.from_pretrained(...)
model.generate(..., use_cache=True)
```




## How to fine-tune an LLM

- [trl RL example](https://huggingface.co/blog/trl-peft)
  - Fine tune a 20B GPT model on text generation on IMDB dataset (loaded in 8 bit)
  - Since step 1 used PEFT, we need to merge the adapter weights with the base model
  - Finally, use RLHF to generate positive movie reviews. They used a BERT IMDB sentiment classifer to generate rewards
- [DataCamp example](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
  - Using `SFTTrainer` from the `trl` library to do supervised fine-tuning
- [PEFT - based on LORA](https://github.com/huggingface/peft) - PEFT is built by hugginface to support LORA.

