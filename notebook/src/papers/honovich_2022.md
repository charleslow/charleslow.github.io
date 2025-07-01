# Honovich 2022 - Instruction Induction

[Instruction Induction: From Few Examples to Natural Language Task Descriptions](https://arxiv.org/abs/2205.10782)

This is an early paper that suggests taking a random subset of training `(input, label)` pairs, showing them to an LLM, and asking the LLM to guess the instruction that produces the label from the inputs.

## Method

For example, the LLM prompt may look something like:
```bash
Here are the input-output pairs:

Input: As soon as you can.
Output: At your earliest convenience.
... (+4 more input / output pairs)

The instruction was ...
```

The LLM may then guess something like `The instruction was translate the inputs into more formal language`. We can then use this instruction as the new optimized prompt.

Note that this method is a one step process, i.e. it does not iterate for further improvements. But I suppose we can sample multiple random subsets of training instances and generate prompts for each before picking the best prompt based on execution accuracy on the validation set.

## Evaluation

The paper found that the execution accuracy of these prompts with instruction fine-tuned version of `GPT-3` (called `InstructGPT`) could reach similar levels to human-generated prompts on most simple tasks.

Notably, the authors were careful to control for the variation caused by the selection of few-shot examples:
- An `induce set` of examples are held out from the training set
- For each run, `5` input / output pairs are randomly drawn from the induce set, and used to generate a prompt
- The evaluation accuracy of this prompt is recorded
- The evaluation accuracy is averaged over `100` runs

In hindsight, by using the idea of [self-consistency](./wang_2022.md), the performance can likely be significantly improved by taking the <<majority vote>> over the `100` runs rather than the average accuracy.

