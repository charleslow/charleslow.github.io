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

## Gemma Models

References:
- [Huggingface Reference](https://huggingface.co/blog/gemma)
- [Fine-tuning Gemma with PEFT](https://huggingface.co/blog/gemma-peft)
- [gemma-7b Model Card](https://huggingface.co/google/gemma-7b)

Google released open source small models under the Gemma label. It comes in two sizes: 2B and 7B parameters, each with base (pretrained) and instruction-tuned versions. All have a context length of `8k tokens`. Gemma integration is supported with `transformers==4.38`. Note that user consent is needed to access Gemma models, so one needs to create a HuggingFace account and [create an access token](https://huggingface.co/docs/hub/security-tokens).

### Sample Code

```python
# Use model_name="google/gemma-7b-it"
def load(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # nested quantization
            bnb_4bit_quant_type="nf4",  # normalized quantization
        ),
        token=TOKEN,
    )
    return model, tokenizer
```

### Inference Speed

Tested out the inference speed of the model using the following prompt: `Can you recommend good alternative occupations for a occupation_name? Please be concise`. 

An example is as follows, for an occupation `counsellor`:

```
Sure, here are some alternative occupations for a counsellor:

* **Social worker:** Similar to counselling, social workers help individuals and families facing various challenges.
* **Community organizer:** Organizes and leads community initiatives to address social issues.
* **Teacher:** Can be a good fit for those who enjoy working with children and adolescents.
* **Human resource manager:** HR managers handle employee relations and conflicts.
* **Public relations specialist:** Helps organizations communicate effectively with the public.
```

The inference speed is about `8-10 seconds` with 4-bit quantization.

### Gemma-2b

Testing out `google/gemma-2b-it` shows that it has poorer performance but faster inference speed at 2 billion parameters. Here is an example of the results for the same prompt.

```
* **Life coach**
* **Coach in education**
* **Career coach**
* **Executive coach**
* **Leadership coach**
* **Management consultant**
* **Business development consultant**
* **Financial advisor**
* **Investment banker**
```

The inference speed is as follows:
- Single: `3s`
- Batch of `4`: `14s`
- Batch of `8`: `16s`
- Batch of `16`: `20s`
- Batch of `32`: OOM on T4 GPU

## Phi-2

Mobius Labs fine-tuned the `phi-2` model from Microsoft which seems promising, and released it under `mobiuslabsgmbh/aanaphi2-v0.1`. The output seems better than `gemma-2b-it`.

```
1. Social worker
2. Mental health therapist
3. School counselor
4. Employee assistance program (EAP) specialist
5. Rehabilitation counselor
6. Family therapist
7. Substance abuse counselor
8. Career counselor
9. Trauma-focused therapist
10. Child and adolescent therapist

These occupations involve working with individuals, families, and communities to promote mental health and well-being, and may provide similar skills and experiences to those of a counsellor.
```

- [bitsandbytes example notebook from HuggingFace](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf#scrollTo=VPD7QS_DR-mw)

Flash attention, torch.compile, quantization.



