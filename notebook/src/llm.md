# Large Language Models

LLMs are generally used in an auto-regressive way, where the user supplies a prompt and the LLM returns a generated response. This framework makes it amenable to a wide range of tasks.

HuggingFace has a great [blog post](https://huggingface.co/blog/optimize-llm) explaining how we can run LLMs on humble hardware. Typically, LLMs have billions of parameters. The following rules of thumb helps us know how much memory we need to load the LLM into memory:
- Loading a model with `X billion` parameters requires `4 * X GB of VRAM` in `float32`
- Loading a model with `X billion` parameters requires `2 * X GB of VRAM` in `bfloat16`

Hence we see that we can load a `~7 billion` parameters model with around `14GB` of VRAM if loaded in `bfloat16`, which makes it feasible to run on GPUs like Tesla T4 with `16GB` of VRAM. This can be done when loading the model with `from_pretrained(..., torch_dtype=torch.bfloat16)`. Most models are trained in `bfloat16` anyway, so it makes sense to load them at that precision.

Current popular open source LLM of that size includes `mosaicml/mpt-7b`, which can be easily downloaded and used using huggingface.

## Quantization

It turns out that we can lower the precision of models even further than 16 bits if we use a quantization method (see e.g. [Dettmers 2022](https://arxiv.org/abs/2208.07339), this paper is the basis for the package `bitsandbytes` used for quantization). The general idea is akin to encoding - we encode each number from a higher precision into a "codeword" in the lower precision (i.e. *quantization*). Numbers that are close to one another in the higher precision may get mapped to the same "codeword". When we want to use the encoded value, we look up the value in the higher precision that it maps to (i.e. *de-quantization*).

When applying this to quantizing a neural network, the steps involved are:
1. Quantize all model weights to target precision (e.g. `int8`)
2. Pass the input vector at `bfloat16`
3. At each layer, dequantize the weights and perform matmul in `bfloat16`
4. Quantize the weights again for storage

Hence while quantization lowers the memory footprint of the model, it may increase inference time. To use quantization, we need to do the following (also make sure `bitsandbytes` is pip installed). We can also pass `load_in_4bit=True` for 4bit quantization. More info on quantization usage is available at [HuggingFace](https://huggingface.co/docs/transformers/main_classes/quantization#general-usage).
```python
model = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)
```

## Flash Attention

The self-attention mechanism ([Dao 2022](https://arxiv.org/abs/2205.14135)) is at the heart of the transformer neural network. Suppose we have an input sequence of embeddings $X = (x_1, ..., x_N)$ where $x_i \in \mathbb{R}^d$, such that $X \in \mathbb{R}^{d \times N}$. Naively, we can compute activations by $V = W_v \cdot X$, where $W_v \in \mathbb{R}^{k \times d}$, such that $V \in \mathbb{R}^{k \times N}$.

## How to fine-tune an LLM

- [trl RL example](https://huggingface.co/blog/trl-peft)
  - Fine tune a 20B GPT model on text generation on IMDB dataset (loaded in 8 bit)
  - Since step 1 used PEFT, we need to merge the adapter weights with the base model
  - Finally, use RLHF to generate positive movie reviews. They used a BERT IMDB sentiment classifer to generate rewards
- [DataCamp example](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
  - Using `SFTTrainer` from the `trl` library to do supervised fine-tuning
- [PEFT - based on LORA](https://github.com/huggingface/peft) - PEFT is built by hugginface to support LORA.

