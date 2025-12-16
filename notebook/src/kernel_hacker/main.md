# The "Kernel Hacker" Roadmap

Roadmap for learning and understanding LLM internals, unsloth optimizations, customized triton kernels etc.

## Phase 1: The New Tooling (Triton & Block Pointers)

> Shift from CUDA C++ mental models to Triton's "Block-Centric" Python.

Learn how triton abstracts tiling and memory coalescing etc.

  * **The Concept:** In CUDA, you calculate `idx` per thread. In Triton, you calculate pointers for a whole *block* of data.
  * **Key Mechanic:** **`tl.make_block_ptr`**. This is the modern way to handle 2D tiling without manual boundary checks inside the hot loop.

### 📝 Action Items

1.  **Tutorial:** Complete [Triton Tutorial 03 (Matrix Multiplication)](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
2.  **Challenge:** Rewrite the MatMul tutorial using `make_block_ptr` (if the tutorial uses the old offset arithmetic).
3.  **Deep Dive:** Read the docs for `tl.load(boundary_check=...)`. Understand how Triton handles padding automatically when your tile goes off the edge of the tensor.

-----

## Phase 2: Operator Fusion (The "Liger" Pattern)

> *Learn to fuse memory-bound operations to reduce HBM reads/writes.*

This is the easiest area to make immediate gains (e.g., Unsloth's 30% speedup on non-attention layers).

  * **The Target:** `RMSNorm`, `CrossEntropyLoss`, `GeLU`.
  * **The "Hack":** Instead of `Read -> Softmax -> Write -> Read -> Log -> Write`, you keep data in SRAM and do `Read -> Softmax -> Log -> Write`.
  * **Mathematical Hurdle:** **"Online Softmax"**. You cannot see the whole row at once to calculate the sum for the denominator. You must calculate running max/sum statistics as you load tiles.

### 📝 Action Items

1.  **Study:** Clone [**Liger-Kernel**](https://github.com/linkedin/Liger-Kernel). Read `liger_kernel/ops/rms_norm.py`. It is cleaner than Unsloth’s code.
2.  **Code:** Implement a Fused **RMSNorm** in Triton.
    * *Steps:* Load vector $\rightarrow$ Cast to FP32 $\rightarrow$ Square $\rightarrow$ Mean $\rightarrow$ Sqrt $\rightarrow$ Cast back $\rightarrow$ Store.
3.  **Verify:** Validate your kernel output against `torch.nn.RMSNorm`.

-----

## Phase 3: The Engine (Attention & Flash)

> *Mastering the inverted loop and memory hierarchy.*

* **The Core Insight:** Standard attention materializes an $N \times N$ matrix. Flash Attention never materializes it.
* **The Algorithm (Inverted Loop):**
  1.  Load a block of Query ($Q$) into SRAM.
  2.  Loop over blocks of Key/Value ($K, V$) from HBM.
  3.  Compute scores and update the output accumulator using online softmax logic.
* **FA2 vs FA3:**
    * **FA2:** Parallelizes over sequence length; uses standard Tensor Cores.
    * **FA3 (Hopper H100 only):** Uses **Warp Specialization** (Producer warps load data, Consumer warps do math) and asynchronous TMA (Tensor Memory Accelerator) copies.

### 📝 Action Items

1.  **Read:** [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691) (Focus on the Algorithm pseudocode, not the proofs).
2.  **Tutorial:** [Triton Tutorial 06 (Fused Attention)](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
      * *Crucial:* Annotate the code where `m_i` (running max) and `l_i` (running sum) are updated. This is the heart of the algorithm.

-----

## Phase 4: LLM Specifics (RoPE, Packing, Precision)

> *Implementing the mechanisms that make modern LLMs work.*

### A. Rotary Positional Embeddings (RoPE)

  * **The Insight:** RoPE is a rotation in a complex plane. It is **memory-bound**.
  * **Implementation:** You don't use matrix multiplication. You apply element-wise rotation:
    $$x_{new} = x_{real} \cdot \cos\theta - x_{imag} \cdot \sin\theta$$
  * **Optimization:** In optimized kernels (Unsloth/FA), RoPE is applied *inside* the attention kernel after loading $Q/K$ to SRAM, but before the dot product.

### B. Sequence Packing (Effective Batching)

  * **The Problem:** Padding tokens waste compute.
  * **The Solution:** Flatten batch into 1D stream `[Total_Tokens, Dim]`.
  * **The Implementation:**
      * **`cu_seqlens`:** A tensor `[0, 100, 550, ...]` defining document boundaries.
      * **Block Diagonal Masking:** Your kernel uses `cu_seqlens` to know when to stop attending (i.e., don't let Doc A attend to Doc B).
      * *Exercise:* Look at `flash_attn_varlen_func` in the Flash Attention repo to see how `cu_seqlens` is passed.

-----

## Phase 5: The "Hacker" Skill (Integration & Monkey Patching)

> *How Unsloth actually works.*

Unsloth does not rewrite the model source code. It modifies the Python objects in memory at runtime.

### 📝 Action Items

1.  **The Pattern:** Write a script to "Patch" a model.
    ```python
    import torch
    from transformers import AutoModelForCausalLM

    # 1. Your Custom Wrapper
    def my_fast_forward(self, x, ...):
        return my_triton_kernel(x)

    # 2. The Patch
    model = AutoModelForCausalLM.from_pretrained("llama-3-8b")
    # Replace the method on the class or the instance
    model.model.layers[0].self_attn.forward = my_fast_forward.__get__(
        model.model.layers[0].self_attn, 
        type(model.model.layers[0].self_attn)
    )
    ```
2.  **Flex Attention:** If writing raw kernels is too hard, learn `torch.nn.attention.flex_attention`.
      * Write a Python `score_mod` that implements a custom mask (e.g., "Sliding Window") and compile it.

-----

## 📚 Reference Library

| Topic | Resource | Why use it? |
| :--- | :--- | :--- |
| **Foundations** | **PMPP Book** | It explains the "Why" (Memory hierarchy). |
| **Fusion Examples** | **[Liger-Kernel](https://github.com/linkedin/Liger-Kernel)** | Readable, production-grade Triton kernels for Norms/Loss. |
| **Attention Code** | **[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)** | The "Hello World" for GPU programming. |
| **Architecture** | **[Flash Attention Repo](https://github.com/Dao-AILab/flash-attention)** | The "Source of Truth" for how packing (`varlen`) is handled. |
| **Hacking** | **[Unsloth Source](https://github.com/unslothai/unsloth)** | Study `models/llama.py` to see how they inject kernels. |