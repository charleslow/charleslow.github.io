# Fine-tuning

LLMs are typically trained with next-token prediction task on large amounts of text in an unsupervised manner. Some of the behaviour in these texts are not desirable to imitate. For example, while Github is full of code repositories with common programming mistakes, we do not want the LLM to replicate such behaviour. Hence, a process of alignment is necessary to encourage the model to produce desired responses.

There are typically two stages to this fine-tuning: Supervised Fine-Tuning and Reinforcement Learning from Human Feedback.

<Supervised Fine-Tuning (SFT)>. In this stage, pairs of `(prompt, desired response)` are provided to the LLM. The desired responses are often called "demonstrations" by human annotators. Some form of cross-entropy loss is then used to update the LLM to encourage it to generate the desired response. This is a straightforward approach that is similar to the next-token prediction task (except we are predicting the desired response given the prompt). In the [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) paper, the authors show that an SFT-aligned `1.3B` model (using ~`13k` training data) generates human-preferred outputs compared to a `175B` GPT model, showing the importance of SFT.

The problem with SFT is that it trains the model to provide a very specific response to a particular prompt, which makes it hard for the model to generalize. It also fails to express the natural ambiguity of responses for an open-ended prompt (e.g. `write a poem about AI`). Hence, unless we can collect quality responses for a very wide variety of prompts, SFT is limited in its ability to generalize.

<Reinforcement Learning from Human Feedback (RLHF)>. This is where RLHF comes in: given triplets of `(prompt, preferred response, not preferred response)`, we train the model to generate the preferred response in a more generalizable way.

## RLHF

Here we trace the derivations from the [DPO paper](https://arxiv.org/pdf/2305.18290.pdf). Denote the model after SFT as $\pi^{SFT}(y | x)$, i.e. the policy $\pi^{SFT}$ is a probability function for each pair of inputs and answers `(x, y)`. Naturally, we can use this policy to generate tokens by choosing the response with the highest probability (or approximate it in a greedy token-by-token manner).

To perform RLHF, we first need to build a <reward model>. First, we prompt the SFT model to obtain pairs of answers $(y_1, y_2) \sim \pi^{SFT}(y\ |\ x)$. These samples are presented to human labellers who record their preferences in the form $y_w \triangleright y_l\ |\ x$, where $y_w$ wins and $y_l$ loses. These preferences are assumed to be generated from an underlying reward model $r^*(x, y)$ which we do not have access to.

We wish to learn this reward model. Since we only have access to pairwise preferences instead of the reward score, a common approach is to model the pairwise preferences using the [Bradley-Terry](../misc/bradley-terry.md) model. Specifically, we assume that the observed human preference decisions are related to the underlying human reward model in the following way:

$$
\begin{align*}
p^*(y_1 \triangleright y_2\ |\ x) &= \frac{e^{r^*(x,\ y_1)}}{e^{r^*(x,\ y_1)} + e^{r^*(x,\ y_2)}} & (1)
\end{align*}
$$

Suppose we have a static dataset of comparisons $\mathcal{D} = \{\ x^{(i)}, y_w^{(i)}, y_l^{(i)}\ \}^N_{i=1}$ sampled from the human preference model of $p^*$. We can create a reward model $r_\phi(x,\ y)$ and use the BT-model to express the negative log likelihood of $\mathcal{D}$. Note that are we using [the re-parametrization of the BT-model as a sigmoid function](../misc/bradley-terry.md#re-parametrization). With this NLL expression, we can optimize for $r_\phi$ using gradient descent to learn the reward model from $\mathcal{D}$.

$$
\begin{align*}
\mathcal{L}(r_\phi,\ \mathcal{D}) &= -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\ log\ \sigma(r_\phi(x,\ y_w) - r_\phi(x,\ y_l)) \ \right] & (2)
\end{align*}
$$

Note that $r_\phi$ is usually initialized using the SFT model $\pi^{SFT}$, but with an additional linear layer over the final transformer layer to generate a scalar value as the reward.

Having learned the reward model, we then need to use the learned reward function to <fine-tune $\pi^{SFT}$ using reinforcement learning>. Specifically, we set $\pi_{ref} := \pi^{SFT}$ as the reference model, and initialize a new $\pi_\theta$ model that we wish to train. Usually, $\pi_\theta$ is also initialized as a copy of $\pi^{SFT}$. The objective function we wish to maximize is:

$$
\begin{align*}
&\max_{\pi_\theta} 
    \mathbb{E}_{\ x \sim \mathcal{D},\ y \sim \pi_\theta(y|x)} 
    \quad r_\phi(x,\ y) 
    - \beta \cdot \mathbb{D}_{KL} 
    \left[  
        \pi_\theta(y\ |\ x)\ ||\ \pi_{ref}(y\ |\ x)
    \right] & (3)
\end{align*}
$$

Inspecting this objective, we see that we are trying to tune $\pi_\theta$ such that it generates answers that maximize the learned reward function $r_\phi$, while at the same time ensuring that we do not deviate too far from the original reference model. $\beta$ is a hyperparameter controlling the degree of deviation. This penalty constraint serves to:
1. Ensure that we do not drift too far from the `(x, y)` distribution on which the reward model is accurate
2. Ensure that we maintain generational diversity and not just collapse into a single high-reward answer for a given prompt

Objective (3) may not be directly optimized, because we need to generate $y \sim \pi_\theta(y\ |\ x)$ at each step from the current policy (not sure if I fully understand this). Hence typically this is optimized using reinforcement learning using PPO.

## Direct Preference Optimization

The RLHF process in general is unstable and requires tricks to make it work. Hence the authors of DPO set out to create an optimization procedure that:
1. Avoids fitting an explicit, standalone reward model
2. Avoids using reinforcement learning

DPO starts off with the KL-constrained reward maximization objective from Equation (3) above. The first step is to show that the optimal policy for this objective is of the form:

$$
\begin{align*}
\pi_r(y\ |\ x) &= \frac{1}{Z(x)} \pi_{ref}(y\ |\ x)\ exp\ \left( \frac{1}{\beta} r(x,\ y) \right) & (4)
\end{align*}
$$

 

## References

- [Ouyang 2022 - InstructGPT](https://arxiv.org/pdf/2203.02155.pdf)
- [Rafailov 2023 - Direct Preference Optimization](https://arxiv.org/pdf/2305.18290.pdf)
- [Huggingface RLHF Blog](https://huggingface.co/blog/rlhf)