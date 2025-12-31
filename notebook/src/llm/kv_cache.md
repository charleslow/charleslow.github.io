# KV-Cache

This article walks through a simple but rigorous demonstration of KV caching in transformer autoregressive decoding.

## The Attention Mechanism

In autoregressive generation, we generate tokens one at a time. Given a sequence of input embeddings $[x_1, x_2, ..., x_t]$, we feed the embedding matrix of size $B \times L \times d$ into the model to predict $x_{t+1}$. Note that:
- $B$ denotes batch size
- $L$ denotes sequence length
- $d$ denotes model dimension
- $V$ denotes output vocabulary size

We will focus on the self-attention block (where the KV-cache is pertinent). For exposition, we will omit the batch size dimension for now and consider a single batch. The self attention block computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q = X \cdot W_Q$ (queries)
- $K = X \cdot W_K$ (keys)  
- $V = X \cdot W_V$ (values)

The transformer produces an output of shape $L \times d$ at the last layer. The output at the final position (call it $h_t^{[L]} \in \R^{d}$) is finally used to generate the logits for each token to predict like so:
$$
    \text{logits} = h_t^{[L]} \cdot W_{out} \in \R^{V}
$$


### Worked Example

Let's make this concrete. Suppose we have:
- 2 tokens already: $x_1, x_2$ (each is a 3-dimensional embedding vector)
- Projection dimension $d_k = 2$

Let's setup our input and projection matrices. Notice that each *row* of $X$ corresponds to one position in the sequence.
$$
X = \begin{bmatrix} — x_1 — \\ — x_2 — \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \quad
W_Q, W_K, W_V \in \mathbb{R}^{3 \times 2}
$$

**Step 1: Generate token 3** (predict $x_3$ given $[x_1, x_2]$)

We compute Q, K, V for all positions:
$$
Q = X \cdot W_Q = \begin{bmatrix} — q_1 — \\ — q_2 — \end{bmatrix}, \quad
K = X \cdot W_K = \begin{bmatrix} — k_1 — \\ — k_2 — \end{bmatrix}, \quad
V = X \cdot W_V = \begin{bmatrix} — v_1 — \\ — v_2 — \end{bmatrix}
$$

Let's write out the full attention computation (omitting softmax and scaling for clarity):

$$
\tilde{A} = Q \cdot K^T = \begin{bmatrix} 
q_1 \cdot k_1 & q_1 \cdot k_2 \\
q_2 \cdot k_1 & q_2 \cdot k_2
\end{bmatrix}
$$

After applying the causal mask (setting future positions to $-\infty$) and softmax, we get attention weights $A$:
$$
A = \begin{bmatrix} 
a_{11} & 0 \\
a_{21} & a_{22}
\end{bmatrix}
$$

Importantly, observe that the bottom row of $A$ only depends on $q_2, k_1$ and $k_2$. $q_1$ is not used at all. We will see that $q_1$ is in fact not needed to generate the next token shortly.

Finally, the output is $O = A \cdot V$:
$$
O = 
\begin{bmatrix} 
a_{11} & 0 \\
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix} v_{11} & v_{12} \\ v_{21} & v_{22} \end{bmatrix}
= \begin{bmatrix} 
a_{11} v_{11} & a_{11} v_{12} \\
a_{21} v_{11} + a_{22} v_{21} & a_{21} v_{12} + a_{22} v_{22}
\end{bmatrix}
$$

To predict $x_3$, we only need the bottom row of $O$. But this accordingly only uses the bottom row of $A$, which only depends on $q_2$. Hence, we can simplify the computation of the bottom row of $O$ as follows:
$$
o_2 = \text{softmax}(q_2 \cdot K^T) \cdot V
$$

**Summary**: To decode for step $t$, we need:
- Only $q_2$ (the query for the last position)
- All of $K_{1:2}$
- All of $V_{1:2}$

Now suppose we want to generate token $4$. We will need:
- $Q_3$
- $K_{1:3}$
- $V_{1:3}$

> <<Key Insight>>: Due to the causal mask, $K_{1:2}$ and $V_{1:2}$ did not change. Hence we can cache them and just append $K_3$ and $V_3$ to get $K_{1:3}, V_{1:3}$. This is the main idea of the <<KV Cache>>.

## Naive Implementation (No Cache)

Without caching, at each generation step we recompute K and V for the entire sequence:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def naive_attention(X, W_q, W_k, W_v):
    """
    X: (batch, seq_len, d_model) - full sequence
    Returns: (batch, seq_len, d_model)
    """
    Q = X @ W_q  # (batch, seq_len, d_k)
    K = X @ W_k  # (batch, seq_len, d_k)
    V = X @ W_v  # (batch, seq_len, d_v)
    
    d_k = K.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Causal mask: prevent attending to future tokens
    seq_len = X.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    return attn @ V

def naive_generate(prompt_embeddings, W_q, W_k, W_v, W_out, num_tokens):
    """
    Generate tokens without KV cache.
    Each step recomputes attention over the ENTIRE sequence.
    """
    X = prompt_embeddings  # (1, prompt_len, d_model)
    
    for step in range(num_tokens):
        # Recompute attention for ALL positions (wasteful!)
        attn_out = naive_attention(X, W_q, W_k, W_v)
        
        # Get logits for last position only
        last_hidden = attn_out[:, -1, :]  # (1, d_model)
        logits = last_hidden @ W_out  # (1, vocab_size)
        
        # Sample next token (simplified: just argmax)
        next_token_id = logits.argmax(dim=-1)
        
        # Embed next token and append to sequence
        # (In practice, look up embedding; here we simulate)
        next_embedding = torch.randn(1, 1, X.shape[-1])
        X = torch.cat([X, next_embedding], dim=1)
        
        print(f"Step {step+1}: Computed K,V for {X.shape[1]} positions")
    
    return X
```

**Complexity without cache**: For generating $T$ new tokens with prompt length $P$:
- Step 1: Compute K,V for $P+1$ positions
- Step 2: Compute K,V for $P+2$ positions
- ...
- Step T: Compute K,V for $P+T$ positions

Total K,V computations: $\sum_{i=1}^{T}(P+i) = T \cdot P + \frac{T(T+1)}{2} = O(T^2 + TP)$

## With KV Cache

The KV cache stores previously computed keys and values. At each step, we only compute K,V for the **new token** and concatenate with the cache:

```python
def cached_attention(x_new, W_q, W_k, W_v, kv_cache):
    """
    x_new: (batch, 1, d_model) - ONLY the new token
    kv_cache: tuple of (K_cache, V_cache) or None
        K_cache: (batch, prev_seq_len, d_k)
        V_cache: (batch, prev_seq_len, d_v)
    
    Returns: attention output, updated cache
    """
    # Compute Q, K, V for the NEW token only
    q_new = x_new @ W_q  # (batch, 1, d_k)
    k_new = x_new @ W_k  # (batch, 1, d_k)
    v_new = x_new @ W_v  # (batch, 1, d_v)
    
    if kv_cache is None:
        K = k_new
        V = v_new
    else:
        K_cache, V_cache = kv_cache
        # Append new K,V to cache
        K = torch.cat([K_cache, k_new], dim=1)  # (batch, seq_len, d_k)
        V = torch.cat([V_cache, v_new], dim=1)  # (batch, seq_len, d_v)
    
    # Attention: new query attends to ALL keys
    d_k = K.shape[-1]
    scores = (q_new @ K.transpose(-2, -1)) / (d_k ** 0.5)  # (batch, 1, seq_len)
    
    # No causal mask needed: q_new is at position seq_len,
    # and we only have K,V up to seq_len (no future tokens)
    attn = F.softmax(scores, dim=-1)
    out = attn @ V  # (batch, 1, d_v)
    
    # Return output and updated cache
    return out, (K, V)

def cached_generate(prompt_embeddings, W_q, W_k, W_v, W_out, num_tokens):
    """
    Generate tokens WITH KV cache.
    
    Two phases:
    1. Prefill: Process entire prompt at once, build initial cache
    2. Decode: Generate one token at a time, updating cache incrementally
    """
    # === PREFILL PHASE ===
    # Process entire prompt to build initial KV cache
    X = prompt_embeddings  # (1, prompt_len, d_model)
    K_cache = X @ W_k  # (1, prompt_len, d_k)
    V_cache = X @ W_v  # (1, prompt_len, d_v)
    
    # Compute attention for prompt (need full Q for all positions)
    Q = X @ W_q
    d_k = K_cache.shape[-1]
    scores = (Q @ K_cache.transpose(-2, -1)) / (d_k ** 0.5)
    seq_len = X.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    attn_out = attn @ V_cache
    
    last_hidden = attn_out[:, -1:, :]  # Keep dim for next iteration
    kv_cache = (K_cache, V_cache)
    
    print(f"Prefill: Built cache with {K_cache.shape[1]} positions")
    
    # === DECODE PHASE ===
    for step in range(num_tokens):
        # Get logits from last hidden state
        logits = last_hidden.squeeze(1) @ W_out
        next_token_id = logits.argmax(dim=-1)
        
        # Embed next token (simulated)
        next_embedding = torch.randn(1, 1, X.shape[-1])
        
        # Compute attention with cache - only process NEW token!
        last_hidden, kv_cache = cached_attention(
            next_embedding, W_q, W_k, W_v, kv_cache
        )
        
        print(f"Step {step+1}: Computed K,V for 1 position (cache size: {kv_cache[0].shape[1]})")
    
    return kv_cache
```

**Complexity with cache**: For generating $T$ new tokens with prompt length $P$:
- Prefill: Compute K,V for $P$ positions (once)
- Each decode step: Compute K,V for 1 position

Total K,V computations: $P + T = O(P + T)$

## Comparison

| Aspect | Without Cache | With Cache |
|--------|--------------|------------|
| K,V computations per step | $O(\text{seq\_len})$ | $O(1)$ |
| Total K,V computations | $O(T^2 + TP)$ | $O(P + T)$ |
| Memory | $O(d)$ | $O((P+T) \cdot d)$ |
| Attention computation | $O(\text{seq\_len}^2)$ | $O(\text{seq\_len})$ per step |

The KV cache trades **memory** for **computation**. This is almost always worthwhile because:
1. Matrix multiplications (K,V projections) are expensive
2. Memory is typically available
3. The speedup is dramatic for long sequences

## Complete Working Example

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Dimensions
batch_size = 1
d_model = 64
d_k = d_v = 32
vocab_size = 100
prompt_len = 5
num_generate = 3

# Initialize weights
W_q = torch.randn(d_model, d_k) / (d_model ** 0.5)
W_k = torch.randn(d_model, d_k) / (d_model ** 0.5)
W_v = torch.randn(d_model, d_v) / (d_model ** 0.5)
W_out = torch.randn(d_v, vocab_size) / (d_v ** 0.5)

# Simulate prompt embeddings
prompt = torch.randn(batch_size, prompt_len, d_model)

print("=== WITHOUT KV CACHE ===")
naive_generate(prompt.clone(), W_q, W_k, W_v, W_out, num_generate)

print("\n=== WITH KV CACHE ===")
cached_generate(prompt.clone(), W_q, W_k, W_v, W_out, num_generate)
```

Output:
```
=== WITHOUT KV CACHE ===
Step 1: Computed K,V for 6 positions
Step 2: Computed K,V for 7 positions
Step 3: Computed K,V for 8 positions

=== WITH KV CACHE ===
Prefill: Built cache with 5 positions
Step 1: Computed K,V for 1 position (cache size: 6)
Step 2: Computed K,V for 1 position (cache size: 7)
Step 3: Computed K,V for 1 position (cache size: 8)
```

## Why Only Cache K and V (Not Q)?

During autoregressive generation:
- **Query (Q)**: We only need the query for the **current** position to compute attention scores. Past queries are never reused.
- **Keys (K)**: The current query must attend to **all** past keys. These are reused every step.
- **Values (V)**: Once we have attention weights, we aggregate **all** past values. These are reused every step.

Mathematically, for generating token at position $t$:

$$\text{out}_t = \text{softmax}\left(\frac{q_t \cdot [k_1, ..., k_t]^T}{\sqrt{d_k}}\right) \cdot [v_1, ..., v_t]$$

Only $q_t$ is new. All $k_i, v_i$ for $i < t$ were computed in previous steps.

## Multiple Layers: Each Layer Has Its Own Cache

Real transformers stack $L$ self-attention layers. **Each layer maintains its own KV cache** because:

1. **Each layer has different projection weights**: Layer $\ell$ has its own $W_K^{(\ell)}$ and $W_V^{(\ell)}$
2. **Each layer receives different input**: The input to layer $\ell$ is the output of layer $\ell-1$

Therefore, even for the same token position, the K and V values differ across layers.

```python
def multi_layer_cached_generate(prompt_embeddings, layers, W_out, num_tokens):
    """
    layers: list of dicts, each with W_q, W_k, W_v for that layer
    
    KV cache structure: list of (K_cache, V_cache) tuples, one per layer
    """
    num_layers = len(layers)
    X = prompt_embeddings  # (1, prompt_len, d_model)
    
    # === PREFILL: Build KV cache for ALL layers ===
    kv_caches = []
    hidden = X
    
    for layer_idx, layer in enumerate(layers):
        W_q, W_k, W_v = layer['W_q'], layer['W_k'], layer['W_v']
        
        # Compute K, V for this layer's input
        K = hidden @ W_k  # (1, prompt_len, d_k)
        V = hidden @ W_v  # (1, prompt_len, d_v)
        kv_caches.append((K, V))
        
        # Compute attention output (with causal mask)
        Q = hidden @ W_q
        d_k = K.shape[-1]
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
        seq_len = hidden.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        hidden = attn @ V  # Output becomes input to next layer
    
    last_hidden = hidden[:, -1:, :]
    print(f"Prefill: Built {num_layers} KV caches, each with {X.shape[1]} positions")
    
    # === DECODE: Update each layer's cache incrementally ===
    for step in range(num_tokens):
        logits = last_hidden.squeeze(1) @ W_out
        next_token_id = logits.argmax(dim=-1)
        next_embedding = torch.randn(1, 1, X.shape[-1])
        
        # Forward through all layers, updating each cache
        hidden = next_embedding
        new_caches = []
        
        for layer_idx, layer in enumerate(layers):
            W_q, W_k, W_v = layer['W_q'], layer['W_k'], layer['W_v']
            K_cache, V_cache = kv_caches[layer_idx]
            
            # Compute K, V for NEW token only at this layer
            q_new = hidden @ W_q  # (1, 1, d_k)
            k_new = hidden @ W_k  # (1, 1, d_k)
            v_new = hidden @ W_v  # (1, 1, d_v)
            
            # Append to this layer's cache
            K = torch.cat([K_cache, k_new], dim=1)
            V = torch.cat([V_cache, v_new], dim=1)
            new_caches.append((K, V))
            
            # Attention: new query attends to all cached K, V
            d_k = K.shape[-1]
            scores = (q_new @ K.transpose(-2, -1)) / (d_k ** 0.5)
            attn = F.softmax(scores, dim=-1)
            hidden = attn @ V  # (1, 1, d_v) - input to next layer
        
        kv_caches = new_caches
        last_hidden = hidden
        
        cache_size = kv_caches[0][0].shape[1]
        print(f"Step {step+1}: Updated {num_layers} caches (each now size {cache_size})")
    
    return kv_caches
```

**Memory for KV cache with $L$ layers**:

$$\text{KV cache size} = 2 \times L \times (\text{seq\_len}) \times d_k \times \text{batch\_size}$$

The factor of 2 is for K and V. For large models (many layers, large $d_k$), this becomes the dominant memory cost during inference.

**Example**: Llama 2 70B has:
- $L = 80$ layers
- $d_k = 128$ per head, 64 heads → 8192 total
- For a 4096-token sequence in fp16: $2 \times 80 \times 4096 \times 8192 \times 2 \text{ bytes} \approx 10.7 \text{ GB}$

This is why KV cache compression techniques (quantization, eviction, etc.) are active research areas.

## References

- [Huggingface Blog: KV Caching](https://huggingface.co/blog/not-lain/kv-caching)