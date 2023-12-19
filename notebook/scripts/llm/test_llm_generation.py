# From https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.bfloat16
)
model.to_bettertransformer()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {
        "role": "assistant",
        "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
    },
    {"role": "user", "content": "Do you have mayonnaise recipes?"},
]

formatted = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = formatted.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
