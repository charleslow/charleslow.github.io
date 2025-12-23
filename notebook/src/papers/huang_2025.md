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

The first part of the script handles dataset loading.

<<load_and_prepare_dataset>> is the main function for data loading and tokenization.
- `data_file`: the `jsonl` file to load
- `tokenizer`: the huggingface tokenizer we are using
- `model_name: str`: the huggingface model we are using
- `predictors`: the number of prediction tokens we will append to the end
- `regular: bool`: If True, do normal LLM SFT, otherwise JEPA. regular is much simpler
- `train_all: bool`: If True, compute loss on all tokens (not just `assistant`)
- `plain: bool`: If True, do not `apply_chat_template`, instead format manually
    - Used when chat template is not available for a particular model
- `front_pred: bool`: If True, put predictor tokens at the start of sequence (not effective).
- `reverse_pred: bool`: If True, swap user and assistant (predict `user` from `assistant`)

Step by step:
- First we use `load_dataset` from huggingface `datasets`
    - Always load the `train` split as train and test are stored in different files
- 


```
Input JSONL:
┌─────────────────────────────────────────┐
│ {"messages": [system, user, assistant]} │
└─────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────┼──────────┐
         │          │          │
         ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌──────────┐
    │  Full  │ │  User  │ │Assistant │
    │  Conv  │ │  Only  │ │  Only    │
    └────────┘ └────────┘ └──────────┘
         │          │          │
         ▼          ▼          ▼
    LM Loss    User Repr   Target Repr
                    └─────┬────┘
                          ▼
                    JEPA Loss
                 (cos similarity)
```