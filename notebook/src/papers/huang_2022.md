# Huang 2022 - LLMs can Self Improve

[Large Language Models Can Self-Improve](https://arxiv.org/abs/2210.11610)

The main idea of this paper is that we can improve the instruction tuning of LLMs for reasoning capabilities using its own synthetic generated data.

## Method

We are given a pre-trained LLM and a question-only training dataset (e.g. like GSM8k). We are also given a few-shot Chain of Thought examples (an example comprises of a question, a reasoning, and a correct answer).

The method is simple. For each question $x_i$ in the training set, we:
- Sample $m$ reasoning paths and answers
- Use <<majority voting>> from the $m$ answers to select the most consistent answer $\tilde{y}$ (this is called self-consistency in the literature). Importantly, to increase diversity:
    - Set the temperature $T > 0$
    - Apply mixed formats of prompts and answers
- Keep all reasoning paths that lead to $\tilde{y}$ as our synthetic dataset
- Fine-tune our LLM on the synthetic dataset using supervised fine-tuning

Note that since we are using self-consistency to obtain "labels" for our synthetic dataset, we do not require training labels.

## Observations

For this method to work, self-consistency needs to be a reliable way of getting accurate answers. The authors plot the confidence score (% of paths leading to $\tilde{y}$) of $\tilde{y}$ against the accuracy at that confidence level, and find that it is highly correlated. This implies that highly consistent answers is a strong indication of correctness.

Generally performance increases as we increase the number of sampled paths $m$. It seems to saturate at around $m=32$. Also, the ideal temperature is around `1.2`, showing that diversity is important for this technique to work well.

### Findings

The fine-tuned LLM significantly advances the SOTA performance:
- GSM8K increases from `74.4` using self-consistency to `82.1` (using self-consistency on the fine-tuned LLM)