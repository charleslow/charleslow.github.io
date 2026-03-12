# Huang 2025 - LLM-JEPA

This paper proposes to add a JEPA loss to regular supervised fine-tuning of LLMs. The JEPA loss tries to minimize distance between the encoding of two pairs of texts that should share the same representation (e.g. `text`: description of a regex, `code`: code representing the regex). 

## Premise

The next token prediction task of LLMs is not very amenable to JEPA-style training. However, there are subtasks that are well suited for JEPA - tasks involving two representations of the same reality. For example, in a "Natural Language to Regex" task, one may view the natural language instruction of `lines containing the string dog two or more times` and the regex `.*(dog)(2,).*` as two views of the same thing.

Normal SFT would just try to predict the regex `code`, token-by-token, from the natural language instruction (`text`). This paper advocates for adding a JEPA-style loss which tries to align the latent representation of the prediction given the input `text` with the latent representation of the `code`.

## Method

Specifically, we can express the usual SFT objective of LLM training as:
$$
    \L_{LLM}(\text{Text}_{1:L-1}, \text{Text}_L) = \text{CrossEntropy}(
        \text{Classifier}(
            \text{Enc}(\text{Text}_{1:L-1}, \text{Text}_L)
        )
    )
$$

That is, the loss is the cross entropy of the next token predicted by our LLM classifier against the tokens of the label. The SFT loss is usually the sum of log losses over the label tokens.

The LLM-JEPA objective adds an additional loss term to the usual SFT loss. The new loss aims to align the predictor representation of the input `text` with the encoder presentation of the `code`.
$$
    \L_{LLM-JEPA} = \sum_{l=2}^L \L_{LLM}(\text{Text}_{1:l-1}, \text{Text}_l) +
        \lambda \cdot d(
            \text{Pred}(\text{Enc}(\text{Text})),
            \text{Enc}(\text{Code})
        )
$$

Some details:
- $\lambda$ is a hyperparameter balancing the contribution of the two terms
- $d$ is the cosine similarity loss in this paper, i.e. $1 - \text{CosSim}(\cdot)$
- <<Encoder>>. The encoding for text and code is taken as the `hidden_state` of the last token from the last layer.
    - To avoid two forward passes, this paper concatenates `text` and `code` to do a single forward pass and extracts the relatively embeddings
    - The `attention_mask` is carefully modified so that `text` and `code` never attend to each other
- <<Predictor>>. Instead of adding more layers, we insert one or more special `[PRED]` tokens (`k` tokens) to the end of the concatenated sequence.
    - These tokens are only allowed to attend to the `text` segment (?)
    - We may view these as enabling additional non-linear computation to process the encoding of the `text` into a latent prediction of the `code`
    - It is interesting because in theory, the model could learn to generate a good latent prediction on these `[PRED]` tokens without affecting its next token prediction. So the fact that this separate task helps the NTP task is non-trivial.
- In sum, there are two forward passes during training <<Note: To check on this part>>:
    - First pass: Over the input + label sequence (with `[PRED]` tokens at the end of the input) with standard causal masking
        - Allows us to compute the usual SFT loss: $\L_{LLM}$
        - Allows us to extract $\text{Pred}(\text{Enc}(\text{Text}))$
    - Second pass: Over the input + label sequence with special blocked masking
        - Allows us to extract $\text{Enc}(\text{Code})$

Finally, because of the increased computational cost, the authors considered dropout to reduce compute and also improve robustness. Specifically, we randomly choose forward passes where only $\L_{LLM}$ is computed. They found that we can have high levels of dropout, e.g. $LD=0.5$ while improving performance and reducing compute cost.

## Findings

The paper spends quite a bit of time exploring the benefits of the JEPA loss. Here we summarize:
- <<LLM-JEPA aligns the encoding of `text` and `code`>>. Specifically, it results in a representation where $\text{Enc}(\text{Code})$ is almost a linear transformation from $\text{Enc}(\text{Text})$. Without the JEPA loss, it is very clearly not the case. Using t-SNE, they also show two nice clusters with JEPA loss.
- <<LLM-JEPA improves fine-tuning accuracy>>. The latent prediction task improves the NTP prediction capability of the LLM as measured by accuracy on downstream tasks. The gains are especially large on the regex synthesis task. 

## Hyperparameters

There are two main hyperparameters to tune, which is not ideal:
- $\lambda$ controls the relative weight between $\L_{LLM}$ and $L_{JEPA}$. The paper searches between $\lambda=1, ..., 4$.
- $k$ controls the number of predictor tokens (representing the computational capacity allocated to the predictor). The paper searches between $k=0, ..., 4$

## Code

Here we do a deep dive into the [code repository for LLM-JEPA](https://github.com/rbalestr-lab/llm-jepa). 

### datasets/

The `datasets/` folder comprises the `train` and `test` splits for different benchmarks stored in `jsonl` files. For example, one line of `gsm8k` looks like below. The aim is to produce the correct answer (exact match) given the system and user prompt.

```json
{"messages": [
    {"role": "system", "content": "Answer the math question, show steps."}, 
    {"role": "user", "content": "James decides to bulk up.  He weighs 120 kg and 
        gains 20% of his body weight in muscle and 1 quarter that much in fat.  
        How much does he weigh now?"
    }, 
    {"role": "assistant", "content": "He gains 120*.2=120*.2=24 24 kg in muscles
    So he gains 24/4=24/4=6 6 kg of fat
    That means he gains a total of 24+6=24+6=30 30 kg of bodyweight
    So his new body weight is 120+30=120+30=150 150 kg
    #### 150"}
    ]
}
```

Note that `messages` has three components: `system`, `user`, `assistant`.

### finetune.py

Recall that the main idea is to add a loss that pushes the representation of the `context` towards the `target`. The main innovation in the code is using attention masking to create <|islands of causal attention|> for efficient computation of these representations.

Before we go into the forward pass, first note that `n` predictor tokens (i.e. `<|predictor_i|>`) tokens are appended to the end of each user message. This simple step allows the network to learn a non-linear prediction function over the user representation, which will be compared against the assistant representation.

#### Naive Approach

Let us start with the naive approach. In the `forward` step, there are normally three tensors to pass:
- `input_ids`: the tokenized ids
- `attention_mask`: the attention mask for causal modelling and masking out padding tokens
- `labels`: the labels which are the inputs shifted right

There are three views of the data:
- Without any suffix: this is the full conversation, which is the concatenation of the user message and the assistant message
- Suffixed by `_user`: this is the user message
- Suffixed by `_assistant`: this is the assistant message

Each view of the data has its own set of attention mask. The labels are set to `-100` for the `_user` and `_assistant` views, since these are used purely for computing the jepa loss and do not contribute to next token prediction loss.

The <|naive approach|> simply treats each view as independent. This means that for a batch size of `B`, it gets transformed into `3B` size:
- The first `B` entries are for the full conversation
- The next `B` entries are for the user view
- The next `B` entries are for the assistant view

After the forward pass, the user and assistant hidden representations will be split up and used to compute the JEPA loss, and added to the NTP loss. Each hidden state for the user / assistant respectively would be `(B, seq_len, hidden_dim)`. The last token embedding is extracted and cosine similarity taken between user and assistant to get JEPA loss.

Clearly, the naive approach creates `3x` the normal forward pass than just using next token prediction. The goal is to use clever masking to use `2x` forward passes instead.

#### Clever Approach

The goal is to align the representation for both user and assistant views, but the representation for both views cannot attend to each other, otherwise there would be data leakage.

The heart of the code lies with a 4D attention mask. First, for a given batch of inputs, it is initialized with:
```py
mask = torch.full(
    (batch_size * 2, 1, seq_length, seq_length), 
    -torch.inf
).to(device)
```

Note that:
- The batch size is doubled because the first half is used for the full conversation's causal mask, and the second half is used for the packed user + assistant mask. 
- The singleton dimension is used to broadcast across attention heads
- We start with `-torch.inf`, so that when added to attention logits before softmax, it will result in `0` attention weights (i.e. masked)
- We start by masking all attention, then we will create gaps subsequently

Next, we compute the last token position for the `user` and `assistant` view respectively:
- The last token index is defined as the last real token before padding begins
- A utility function is used to walk each row (for the user and assistant view respectively) to find this last token

Next, we copy the `input_ids` and `attention_mask` from the assistant message into the user message, such that it starts immediately after the user message ends. This is so that we can do a single forward pass on both.
- After this operation, `input_ids_user[i]` looks like: `[user_tok_0, user_tok_1, ..., user_tok_N, asst_tok_0, asst_tok_1, ..., asst_tok_M, pad, pad, ...]`

Now we punch holes into the attention mask. The exact implementation is not important to know, but an example will suffice. Suppose `length_user=3` and `length_assistant=2`, then the mask for the concatenated sequence along the last two dimensions will look like:
```
       u0   u1   u2   a0   a1
  u0 [  0  -inf -inf -inf -inf ]
  u1 [  0    0  -inf -inf -inf ]
  u2 [  0    0    0  -inf -inf ]
  a0 [-inf -inf -inf   0  -inf ]
  a1 [-inf -inf -inf   0    0  ]
```

We can see that this mask prevents the user and assistant tokens from attending to one another, so the single forward pass will allow us to get the user and assistant representations. After the forward pass, some indexing is done to extract the hidden states of the user and assistant representation respectively. 
