# karpathy/autoresearch

[Code Stories Link: autoresearch-end-to-end](https://charleslow.github.io/code-stories/?repo=charleslow/code-stories-cache&story=autoresearch-end-to-end)

Karpathy's autoresearch is an experiment on agent-led experimentation. The idea is really simple - there are only 3 pertinent files in the repo.

<|program.md|> - A natural language instruction on the goal and structure of the experimentation loop. It reads like a brief to a graduate student:
- Look at the git state of the repo
- Hack `train.py` to test experimental ideas
- Run the experiment and read the results
    - If crash, try to fix a few times and then give up
    - If improved, git commit and start again
    - If deproved, git reset to the original state
- Final instructions:
    - Timeout: each run takes 5 mins total (<|note|>: also enforced in code)
    - Crash: use judgement, fix easy bugs or give up on fundamental issues
    - Never stop: never ask human for input, keep going indefinitely

<|prepare.py|> - Mostly a one-off data preparation script to download and tokenize the data. Also defines some constants that are not changeable, and the evaluation function that computes bits per byte. The agent does not touch this file.
- Important constants:
    - `TIME_BUDGET=300`: 5 min timeout
    - `EVAL_TOKENS`: fixed number of tokens for validation evaluation
    - `MAX_SEQ_LEN=2048`: max sequence length
- Evaluation function:
    - `evaluate_bpb(model, tokenizer, batch_size)` computes cross entropy loss and then normalizes by total bytes of validation tokens
    - Cross entropy loss is expressed in `nats` because `ln` is used
    - Divide by `ln(2)` to convert nats to bits
    - Divide by byte count of validation tokens so that the metric is agnostic to the tokenizer vocabulary size
    - Intuition is that a tokenizer with larger vocab size means that each token can represent longer words, i.e. each token carries more information bits by default
- Dataloading choice:
    - Interesting packing strategy to ensure that each batch of tokens have the same length to minimize padding
    - This is to squeeze out as much information as possible in the limited compute window of 5 mins

<|train.py|> - The file that the agent is supposed to hack. The code is quite sophisticated, filled with various optimization tricks that Karpathy implemented in nanochat. So it is a strong baseline, implementing the entire transformer training loop from pytorch primitives. 

Will not dive into details of `train.py` here (that would be more of a nanochat deep dive), but some design choices:
- The training loop imports the immutable `TIME_BUDGET` from `prepare.py`, and imposes a stop when `total_training_time > TIME_BUDGET`
- `evaluate_bpb` is called on the model and tokenizer to get final `val_bpb` which is printed out at the end of the run

## Takeaway

That is about it! What's surprising to me is also what is **not** in the code: there is no sophisticated harness to ensure that the agent cannot cheat. Technically, there is nothing stopping the agent from removing the time budget from `train.py` or hacking the evaluation function. The takeaway is that <|the prompt is sufficient harness|>, at least for this setup and using claude. 

There is also too much hype about this experiment. The tweaks that the agent makes are mostly hyperparameter changes, such as tweaking various learning rates and RoPE parameters, so it can be seen as a glorified hyperparameter tuning run which is not new. 

Nonetheless, we should not downplay the usefulness of Karpathy's setup:
- Design choice of `5 min` on a H100 and using bits per bytes is a neat setup to force rapid interation. He also demonstrated that a simple setup as above is sufficient to get the agent to comply
- Unlike standard hyperparameter tuning, where the parameter space is pre-defined and limited scope, <|the entire architecture (in theory) is open to hacking|>. This is much more open-ended and the upper bound is unlimited. We will surely find ways to improve upon this structure and find non-trivial things using autoresearch-ish approaches. For a start, I hope to do something similar to get an agent to run through some ML experiments, using the time budget as a way to conduct effective dev runs.