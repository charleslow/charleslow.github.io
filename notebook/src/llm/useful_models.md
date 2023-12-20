# Useful Models

There are too many models on Huggingface, so I try to keep track of useful ones here.

- [distilgpt2](https://huggingface.co/distilgpt2). 88 million parameters, super small model that is feasible to run on CPU. Unfortunately the performance seems quite poor.
- [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom). Architecture of BLOOM is essentially similar to GPT3, but is trained on 46 different languages. The original BLOOM has `176B` parameters, but a [7B version](https://huggingface.co/bigscience/bloom-7b1) exists as well. Current score of the `7B` model on the [openllm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is 42.21.
- [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0). `10.7B` parameters, but with an openllm score of 74.2! This model was trained using Supervised Finetuning (SFT) and Direct Preference Optimization (DPO).
