# Useful Models

There are too many models on Huggingface, so I try to keep track of useful ones here. Generally the [open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is a good place to start.


| Model Name | # Params (Billions) | openllm score | Comments |
|:----------:|:--------:|:-------------:|:--------:|
| [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) | 10.7B | 74.2 | This model was trained using Supervised Finetuning (SFT) and Direct Preference Optimization (DPO) for single-turn chat. Seems to be derived from Llama. Verified that the outputs are sensible. |

## Testing SOLAR-10.7B

Here are some results from testing out `SOLAR-10.7B-Instruct-v1.0` on Google Colab, with a T4 GPU. First we install packages and load the model. Note that we are quantizing it to `4bits` to save memory, and specifying the compute type of `torch.float16` as that is the tensor type that this model was trained in. Using `torch.float32` (which is the default) will incur unnecessary conversion cost and slow down inference.

```python
%pip install transformers~=4.36 torch~=2.1.1 bitsandbytes accelerate optimum

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

MODEL_NAME = "Upstage/SOLAR-10.7B-Instruct-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quantization_config,
)
```

We define a generate function.

```python
def generate(prompt: str, max_length: int = 2048):  
    conversation = [{'role': 'user', 'content': prompt}] 
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, use_cache=True, max_length=max_length)
    output_text = tokenizer.decode(outputs[0])
    return output_text
```

And now we can test it out on a few prompts.

Prompt: `Can you explain web3 to a 5 year old?` Response: Web3 is like a special world on the internet that lets people share and use things in a new and more open way. Imagine a playground where everyone can add their own swings, slides, and sandboxes without needing permission from a big boss. This makes it easier for people to work together and create new fun things online. (Inference time: `8s`)

Prompt: `Can a firefighter transit into the finance industry?` Response: While it may not be a typical career transition, it is possible for a firefighter to transition into the finance industry. To make a successful move, the firefighter would need to acquire the necessary skills and qualifications relevant to the finance sector. This could involve pursuing additional education or training in finance, accounting, or business administration. Networking, gaining relevant work experience, and showcasing transferable skills from the firefighting profession can also help in making the transition.\n\nSome transferable skills from firefighting to finance could include strong problem-solving abilities, teamwork, decision-making under pressure, and the ability to manage risk. However, it's essential to note that the finance industry has its unique requirements, and the firefighter would need to demonstrate a strong commitment to learning and adapting to the new environment. (Inference time: `19s`)

Seems to be pretty decent performance at a somewhat reasonable inference time out of the box.

[Speeding up GPU inference](https://huggingface.co/docs/transformers/perf_infer_gpu_one).