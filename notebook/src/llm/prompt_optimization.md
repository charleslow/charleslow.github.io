# Automatic Prompt Optimization

Automatic prompt optimization is a research area of recent interest. This trend arises from the following observations:
- Adjusting textual prompts can very significantly change LLM's ability to accurately complete tasks
- The correlation between prompt and performance is not always the most obvious or intuitive to humans
- The best prompt varies depending on the LLM model (or even different iterations of the same model)

LLMs are increasingly used to solve diverse problems by varying the instructions. This is often a simpler solution than maintaining many models that do various specific tasks. For example, in the recsys world, companies like [LinkedIn](https://arxiv.org/abs/2501.16450) and [Netflix](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) are moving toward foundational LLM models that can replace a collection of traditional recommender models. By adjusting the instructions to the foundation model, it can achieve comparable or even better performance to these task-specific models due to favourable scaling laws with the size of the model. 

However, manually tuning the instruction for each task is brittle and model performance can easily change as the underlying foundation model is continually trained and improved, or swapped out. Also, we may often want to adapt a foundational model to a new task without performing further fine-tuning or training adapter weights. Hence, automatic prompt optimization becomes an important tool to maintain model performance in an operational setting.

This note focuses on gradient-free methods to optimize black-box LLMs, as they are often simpler and are applicable to LLM usage via API, which is still the most common LLM usage method outside of big tech companies. I also follow [Wolfe 2024](#wolfe2024) heavily but simplify quite a bit for readability.

## Main Idea

The main idea of automatic prompt optimization is to simply treat the task of prompt optimization as a standard machine learning problem. That is, we construct a training dataset and optimize our prompt to improve performance on it, and validate the effectiveness of the prompt on a held out validation set. The only difference is that we have to be creative in how we optimize the prompt, since we cannot use gradient descent to do so.

Specifically, the general setup that we consider is such:
- We have a dataset of inputs and labels that can be split into train and validation sets
    - Labels are optional if we are using another LLM as judge
- For each training example, we can produce textual output from an LLM call: $output = LLM(prompt, input)$
- We have some evaluation function that returns a score for this example instance: $score = Eval(output, label)$
    - The evaluation function obviously depends on the specific task
    - E.g. for simple tasks, the evaluation function can simply be the accuracy of $output == label$
    - The evaluation function can also be another LLM for ambiguous tasks
- We can thus compute the mean score across a set of training instances to evaluate the effectiveness of a given prompt
- At each iteration, we try one or more new prompt(s) and evaluate their performance, then have some way of generating new prompts to try again
- At the end, the best prompt is selected

## Methods

Here we dive into the different methods. Some are quite ingenious in how they use LLMs to elicit the best instructions. Note that we omit papers that perform prompt optimization through some gradient-based optimization (e.g. using reinforcement learning), as these cannot be used with LLMs accessed through API.

### Instruction Induction

[Honovich 2022](#honovich2022) is an early paper that suggests taking a random subset of training `(input, label)` pairs, showing them to an LLM, and asking the LLM to guess the instruction that produces the label from the inputs.

For example, the LLM prompt may look something like:
```bash
Here are the input-output pairs:

Input: As soon as you can.
Output: At your earliest convenience.
...

The instruction was <PLS FILL IN>
```

The LLM may then guess something like `The instruction was translate the inputs into more formal language`. We can then use this instruction as the new optimized prompt.

Note that this method is a one step process, i.e. it does not iterate for further improvements. But I suppose we can sample multiple random subsets of training instances and generate prompts for each before picking the best prompt.







## Bibliography

- <a id="honovich2022"></a>[Honovich 2022: Instruction Induction](https://arxiv.org/abs/2205.10782)
- <a id="yang2023"></a>[Yang 2023: Large Language Models as Optimizers (OPRO)](https://arxiv.org/abs/2309.03409)
- <a id="wolfe2024"></a>[Wolfe 2024: Automatic Prompt Optimization](https://cameronrwolfe.substack.com/p/automatic-prompt-optimization)