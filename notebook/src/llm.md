# Large Language Models

LLMs are generally used in an auto-regressive way, where the user supplies a prompt and the LLM returns a generated response. This framework makes it amenable to a wide range of tasks.

HuggingFace has a great [blog post](https://huggingface.co/blog/optimize-llm) explaining how we can run LLMs on humble hardware. Typically, LLMs have billions of parameters. The following rules of thumb helps us know how much memory we need to load the LLM into memory:
- Loading a model with `X billion` parameters requires `4 * X GB of VRAM` in `float32`
- Loading a model with `X billion` parameters requires `2 * X GB of VRAM` in `bfloat16`

Hence we see that we can load a `~7 billion` parameters model with around `14GB` of VRAM if loaded in `bfloat16`, which makes it feasible to run on GPUs like Tesla T4 with `16GB` of VRAM. This can be done when loading the model with `from_pretrained(..., torch_dtype=torch.bfloat16)`.

Current popular open source LLM of that size includes `mosaicml/mpt-7b`, which can be easily downloaded and used using huggingface.
