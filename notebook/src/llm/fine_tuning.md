# Fine-tuning

LLMs are typically trained with next-token prediction task on large amounts of text in an unsupervised manner. Some of the behaviour in these texts are not desirable to imitate. For example, while Github is full of code repositories with common programming mistakes, we do not want the LLM to replicate such behaviour. Hence, a process of alignment is necessary to encourage the model to produce desired responses.

There are typically two stages to this fine-tuning: Supervised Fine-Tuning and Reinforcement Learning from Human Feedback.

<Supervised Fine-Tuning (SFT)>. In this stage, pairs of `(prompt, desired response)` are provided to the LLM. The desired responses are often called "demonstrations" by human annotators. Some form of cross-entropy loss is then used to update the LLM to encourage it to generate the desired response. This is a straightforward approach that is similar to the next-token prediction task (except we are predicting the desired response given the prompt). In the [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) paper, the authors show that an SFT-aligned `1.3B` model (using ~`13k` training data) generates human-preferred outputs compared to a `175B` GPT model, showing the importance of SFT.

The problem with SFT is that it trains the model to provide a very specific response to a particular prompt, which makes it hard for the model to generalize. It also fails to express the natural ambiguity of responses for an open-ended prompt (e.g. `write a poem about AI`). Hence, unless we can collect quality responses for a very wide variety of prompts, SFT is limited in its ability to generalize.

<Reinforcement Learning from Human Feedback (RLHF)>. This is where RLHF comes in: given triplets of `(prompt, preferred response, not preferred response)`, we train the model to generate the preferred response in a more generalizable way.

More recently, a method called Direct Preference Optimization is used to solve this problem without needing to do reinforcement learning. More on that [here](../papers/rafailov_2023.md).
