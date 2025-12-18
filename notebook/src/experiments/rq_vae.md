# RQ-VAE Fast Decoding

[Github Repo: rq_vae](https://github.com/charleslow/rq_vae)

Experiments on applying RQ-VAE to fast text decoding in latent space.

## Main Idea

We use ideas from two papers:
- [Kaiser 2018: Fast Decoding in Sequence Models using Discrete Latent Variables](https://charleslow.github.io/notebook/book/papers/kaiser_2018.html)
    - Applies the idea of representing tokens in a shorter latent space, and then doing autoregressive text translation in the latent space, then upsample back to token space
    - Still uses old VQ-VAE discretization which has issues
- [Lee 2022: Autoregressive Image Generation using Residual Quantization](https://charleslow.github.io/notebook/book/papers/lee_2022.html)
    - Better way of doing discretization, using a codebook with multiple levels instead of a flat codebook
    - Some tricks of using a specialized transformer for decoding that is faster

## Experiment Goal

Verify if we can achieve significant speedup by decoding in latent space without losing accuracy
- Check perplexity on held out data
- Check codebook usage
- Compare against various base models (including qwen0.6B) on standard LLM tasks

Other goals:
- Test if RQ-transformer matches standard transformer decoding for the same compute
- Test if using a pretrained decoder backbone is necessary
- Test performance - inference speed trade-off as we increase compression factor
- Test codebook vocabulary size to codebook depth tradeoff
- Test if unsloth speeds up inference significantly
- Test if we can scale to 8B models

## Overall Flow

- Start with text input (e.g. `fineweb`)
- Encode into a latent representation that is a shorter sequence
    - Run through pretrained backbone (e.g. `qwen0.6B`)
    - Downsample using convolutions
- Run RQ-VAE on the latent representation to discretize into latent tokens
    - This is the <<fast decoding>> part
- Decode latent tokens back into original textual space
    - Upsample using de-convolutions
    - Run through pretrained backbone (e.g. `qwen0.6B`)

## Training Sequence

The trainable components are encoder, RQ-codebook, decoder for the RQ-VAE portion, and also the RQ-transformer for the latent decoding.
- Train the RQ-VAE jointly
    - Freeze pretrained qwen backbone for warmup
    - Then unfreeze for full training
- Train the RQ-transformer on latent sequences

## Inference Sequence

At inference time, the goal is to do autoregressive text completion (like normal LLMs):
- Given input text
- Use encoder to encode into latent space
- Use RQ-transformer to auto-regressively predict full latent sequence
- Use decoder to decode back into textual space

## Code Deep Dive

Here we dive into dissecting the code in detail. The code structure is:

```
│  model/                                                                  
│  ├── rq_vae.py
│  │   └── RQVAE ─────────────┬─► TextEncoder                              
│  │                          ├─► ResidualQuantizer                        
│  │                          └─► TextDecoder                              
│  ├── encoder.py 
│  ├── decoder.py 
│  ├── quantizer.py
│  ├── rq_transformer.py
│  │   ├── SpatialTransformer
│  │   ├── DepthTransformer
│  │   ├── RoPEAttention
│  │   └── TransformerBlock
│  └── layers.py
│      └── SwiGLUTransformerLayer                                          
│  train_vae.py
│  │   ├── RQVAELightningModule                                            
│  │   └── RQVAEDataModule                                                 
│  train_transformer.py
│      ├── RQTransformerLightningModule                                    
│      └── RQTransformerDataModule                   
```

### encoder.py

Implements `TextEncoder` which encodes an input text into latent embeddings representation (shorter sequence). We focus on the forward pass:
- Input text sequence: `batch_size, seq_len`
- Encode using pretrained `qwen` backbone: `batch_size, seq_len, hidden_size`
- Downsample into shorter latent sequence: `batch_size, compressed_len, hidden_size`
    - Use strided convolutions to downsample the sequence length (halves length each step)
    - More details below
- Linear projection into `batch_size, seq_len, latent_dim`
    - The `latent_dim` is the dimension of our RQ-VAE codebook
- Refine latents using self-attention: `batch_size, seq_len, latent_dim`
    - Do it with `SwiGLUTransformerLayer` for `num_latent_layers` times
- Output the latent representation

> <<Question>>: How much information do we lose in the convolutional downsampling? How can we encourage the latent space to store more information?

Diving a bit more into the convolutional downsampling process:
- `nn.Conv1d` applies a convolutional filter which takes a weighted sum of the input values within the window
    - `kernel_size` is the size of the convolution window / filter. Larger value means we average more values together.
    - `stride` is how many steps we move when sliding the window. `stride=2` approximately halves the sequence length.
    - `padding` is the number of zero-padded cells we add to each side of the input
- For an input sequence of $L_{in}$, padding $p$, kernel_size $k$, and stride $s$, the output length is:
$$
    \left\lfloor
        L_{out} = \frac{
            L_{in} + 2p - k
        }{
            s
        } + 1
    \right\rfloor
$$
- The way to think about the formula is that we start with the first window (hence $+1$ at the end) and then count the number of strides to take
    - First we add padding $2p$ to pad the input size
    - Subtract $k$ because we already "occupy" the first kernel width
    - Add a `floor` operator to account for partial strides which we discard
- Can use the [interactive convolution explorer](./conv_viz.html) to visualize the convolution window

### decoder.py

The decoder `TextDecoder` is very similar to the encoder, except we go in the other direction. From a latent sequence, we upsample using de-convolutions to return to the textual space. Again we focus on the forward pass:
- Input is a sequence of compressed latents: `batch_size, compressed_len, latent_dim`
- Refine latents with self-attention: `batch_size, compressed_len, latent_dim`
- Linear projection to hidden size: `batch_size, compressed_len, hidden_size`
- Upsample via transposed convolutions: `batch_size, seq_len, hidden_size`
- Each layer 2x expands sequence length
- Process through Qwen3 backbone (one-shot, not autoregressive): `batch_size, seq_len, hidden_size`
- Linear projection to vocabulary logits: `batch_size, seq_len, vocab_size`

### quantizer.py

Now we dive into the residual quantization portion. 

The <<Quantizer>> class represents one level of the codebook. It is initialized with parameters:
- `dim`: latent dimension of codebook vectors
- `codebook_size`: number of vectors for this level of the codebook
- `ema_decay`: the rate of updating codebook vectors (analogous to `1 - learning_rate`)
- `threshold_ema_dead_code`: number of dead codes we allow before we re-assign these codebook vectors

We register these buffers at init:
- `ema_cluster_size`: `torch.zeros(codebook_size)`
    - Tracks the number of times each codebook vector is used in an exponential moving average
- `ema_embed_sum`: `torch.zeros(codebook_size, dim)`
    - Tracks the sum of encoder embeddings assigned in an exponential moving average

The `quantize` method assigns an input encoder embedding `x` to the nearest codebook vector. Note that `x` would represent a `residual` vector (depending on the level we are at in the codebook) in the RQ-VAE architecture.
- First we use `torch.cdist` to find distances to all the codebook vectors
- Then we use `torch.argmin` to get the index of the nearest codebook vector
- Then we retrieve the codebook embedding corresponding to this index
- We return a tuple `(quantized, indices)`

The `update_codebook_ema` acts like a gradient update to the codebook vectors (following [Oord 2017](../papers/van_den_oord_2017.md)).
- `one_hot: batch_size, codebook_size = F.one_hot(indices, codebook_size)`
    - After we run `quantize` we convert the nearest neighbour `indices` into a one-hot matrix 
- `cluster_size = one_hot.sum(dim=0)`
    - The one-hot matrix  is summed on dimension `0` to give the number of counts per cluster 
    - Shape is `codebook_size`
- `embed_sum = one_hot.t() @ x`
    - This is the sum of encoder embeddings based on assignment to clusters
    - Shape is `codebook_size, latent_dim`
- The exponential moving average of `cluster_size` and `embed_sum` are taken and assigned to `self.ema_cluster_size` and `self.ema_embed_sum` respectively
    - The codebook vectors are updated as `self.ema_embed_sum / self.ema_cluster_size`
- Dead code checking to avoid codebook collapse
    - Codebook collapse is the case where some codebook vectors are so far away from the encoder distribution that they never get used
    - We detect these codes by tracking their exponential moving average of counts
    - If the average assignment falls below a certain threshold, we delete these codebook vectors and re-initialize them to an average of a few random encoder embeddings in the batch

There is also `compute_perplexity`, which is used to measure the distribution of codebook utilisation using <<perplexity>>
- Perplexity is defined as $\exp(\text{entropy}) = \exp(\sum^K_{i=1} -p_i \log p_i)$ of a categorical distribution with $K$ categories
- We know that entropy ranges from $0$ to $\log K$
- So perplexity ranges from $1$ to $K$
- We want the perplexity to be close to $K$, showing that codebook utilisation is close to uniform

> <<Question:>> 
> - Is uniform codebook utilisation optimal, or is there some way to reason about the ideal distribution of codebook utilisation?
> - There seems to be some reason to reduce the codebook size as we go into deeper levels, since the variance decreases and there is higher risk of modelling noise. Is there some information theoretic way to <<dynamically adjust the codebook size>> based on the amount of information gain from that codebook level?

The <<ResidualQuantizer>> class is a stack of <<Quantizer>>s based on the desired number of `codebook_levels`.

The `forward` pass is where most of the logic resides:
- We receive an input tensor `x` with shape `batch_size, seq_len, latent_dim`
- We run a for loop through all the `codebook_levels`:
    - `quantized, indices = quantizer.quantize(residual)`
        - Run the quantizer to get the assigned codebook vector and indices
    - Update the exponential moving average for the quantizer
    - `commitment_loss = F.mse_loss(residual, quantized.detach())`
        - Compute the commitment loss that pushes encoder output to be near codebook vectors
        - Notice the stop gradient on the codebook vectors
    - Compute perplexity and keep track
    - Accumulate `quantized` into `quantized_out`
        - Recall that RQ-VAE represents each token as a sum of the codebook vectors across all levels
        - Hence we sum `quantized` at each level to get our final representation
        - The straight-through estimator is applied to `quantized_out` to get gradients flowing back to the encoder 


