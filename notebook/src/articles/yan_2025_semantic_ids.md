# Yan 2025 - Semantic IDs

- Article: [https://eugeneyan.com/writing/semantic-ids/](https://eugeneyan.com/writing/semantic-ids/)
- Code: [https://github.com/eugeneyan/semantic-ids-llm](https://github.com/eugeneyan/semantic-ids-llm)

Eugene Yan's experience of training an LLM-recommender hybrid that can recommend items using natural language.

## Extend the vocab

Insert semantic ID tokens like `<|sid_0|>`, `<|sid_1|>` into the vocab. Sequences of these tokens will be used to represent the catalog. So an item may be represented like:
```
<|sid_start|><|sid_173|><|sid_324|><|sid_764|><|sid_768|><|sid_end|>
```

## Data

He used video games category of Amazon Reviews 2023, because it has rich product metadata and lots of behavioural data.
- Keep only products with titles longer that `20 chars` and description longer than `100 chars`
- `66k` products
- `737k` rows of interactions

## Training Semantic IDs

Used RQ-VAE to train semantic IDs. Will not expound on RQ-VAE here. Eugene used:
- `3-level` codebook
- Each level has `256` codes
- This has collisions on around `10%` of the `66k products`
- Added a sequentially increasing token to each ID to ensure uniqueness

## Baselines

1. Train a SASRec model on semantic IDs to compare against traditional SASRec
2. Use Qwen3-Embedding-0.6B to encode product metadata into embeddings
3. Finetune Qwen3-8B to recommend items via semantic IDs

## Data Cleaning

- Clean item descriptions using Gemini 2.5 Flash to remove HTML, reduce verbosity
- Remove promotional text, standardize formatting for titles etc.
- Augment data by extracting structured metadata like product type, platform, genre, hardware type etc.
- Build user interaction histories

## Training RQ-VAE

First, we get the `1,024` embeddings using Qwen3-0.6B, then l2-normalize. The RQ VAE consists of:
- An encoder
- A codebook with 3 quantization levels
- A symmetric decoder

Some tricks:
- Use rotation trick as a replacement for the straight through estimator 
- Initialize codebooks with k means clustering
- Reset unused codes
- Large batch size

Metrics for measuring quality of RQ-VAE:
1. `loss/reconstruction`: How well we can reconstruct the original item embeddings after compressing and decompressing
2. `loss/vq`: Combined codebook and commitment loss across all levels.
- Ensures that encoder outputs and codebook vectors stay close together
3. `loss/total`: sum of `loss/vq` and `loss/reconstruction`
4. `loss/validation`: total loss but on the held out validation set. Our <<deciding metric>>.
5. `metrics/avg_residual_norm`: Leftover residuals between the quantized embedding and the original embedding
6. `metrics/unique_ids_proportion`: % of items with unique IDS in a batch
- Helps to check against codebook collapse
- We want this metric to be high
7. Codebook distribution
- Plot the item distribution amongst the `256` codes at each level
- It should look like a uniform distribution

Some hyperparameter tuning:
- Set $\beta = 0.5$:
    - Tried $\beta = 0.25, 0.5, 1.0`. $\beta$ is the commitment weight that balances reconstruction accuracy with codebook commitment
    - $\beta = 0.5$ had the lowest validation loss, so it was chosen
- Investing in data cleaning significantly improved all the metrics

## SASRec comparison

Eugene tested two SASRec variants:
- Traditional SASRec
    - Each item has a unique ID with no semantic meaning
    - Model uses 2 causal attention blocks, 64 dim hidden dimension, trained with binary cross entropy loss
    - Dot product of embeddings is used to generate scores
- Semantic SASRec
    - For semantic version, each item is a sequence of semantic tokens
    - Hence for each position, the model needs to generate a sequence of tokens to represent an item
    - Instead of binary cross entropy loss, we need to sum up the cross entropy loss at each position
    - <<Question>>: does it make sense to add some weights to put more weightage on the first semantic token and decay it for subsequent positions?
    - Teacher forcing is used for this training
    - A larger model is used here, 4 causal attention blocks and 384 hidden dim

From this experiment, Eugene found that traditional SASRec is significantly better. But he puts this down to the difficulty of generating a sequence of tokens compared to directly generating one token. I also note that we are not using a pretrained LLM here, which means we are missing out on some pretrained capabilities that we could have tapped on.

## Finetuning Qwen-8B

Now finally we train the language model to converse in semantic IDs.

First, we build the training dataset of `4.2 million` conversations of various task types:
- Given a semantic ID, predict the item description
- Given item description, predict the semantic ID
- Predict the next item in the user's sequence
- Understand relationships between item categories
- Multi-hop reasoning

Each of these are formatted as a conversation with a system prompt, user prompt and assistant response. Now, we finetune the model in two phases.

In <<Phase 1>>, we focus on extending the model's vocabulary. 
- We add `1,027` new vocabulary items to the model's vocabulary, and resize the model's embedding matrix for them.
- In this phase, we freeze all the model parameters except the input and output embedding layers, training `15%` of total parameters
- Train for `1,000` steps with relatively high learning rate

In <<Phase 2>>, we do the full training:
- Train for 3 epochs on the full dataset
- 3% warmup, lower learning rate, 8-bit AdamW optimizer for memory efficiency
- Monitor progress with callbacks:
    - Track gradient norms
    - Measure hierarchical accuracy (I guess on different codebook granularities) on recommendation tasks
    - Sample model outputs on fixed set of conversational inputs

## Inference

Now the model is trained, we can have chat / recommendation conversations with it:

> <<User>>: “Halo 3 Limited Edition -Xbox 360”, “Lost Planet: Extreme Condition - Xbox 360”, “Lost Planet: Extreme Condition Collector’s Edition -Xbox 360” <|rec|>
> 
> <<Model>>: `<|sid_start|><|sid_64|><|sid_351|><|sid_733|><|sid_768|><|sid_end|>`

The model can also explain its recs:
> <<User>>: I just finished “Dragon Quest Heroes II Explorer’s Edition - PlayStation 4”. Suggest another <|rec|> and explain why:
> 
> <<Model>>: `<|sid_start|><|sid_139|><|sid_289|><|sid_534|><|sid_768|><|sid_end|>`
> 
> If you liked Dragon Quest Heroes II, you might like Nights of Azure because both are action RPGs for the PlayStation 4 with a focus on combat and character progression. Both games offer a narrative-driven experience with a strong emphasis on combat mechanics, suggesting a shared player interest in this genre.

We can even have multi-turn conversation with the model:
> <<User>>: I’m looking for games similar to Mario Kart. <|rec|>
> 
> <<Model>>: `<|sid_start|><|sid_131|><|sid_503|><|sid_738|><|sid_768|><|sid_end|>, <|sid_start|><|sid_131|><|sid_260|><|sid_577|><|sid_768|><|sid_end|>, <|sid_start|><|sid_145|><|sid_416|><|sid_671|><|sid_768|><|sid_end|>`
> 
> <<User>>: How about something similar but for Xbox? <|rec|>
> 
> <<Model>>: `<|sid_start|><|sid_183|><|sid_461|><|sid_517|><|sid_768|><|sid_end|>, <|sid_start|><|sid_183|><|sid_461|><|sid_562|><|sid_768|><|sid_end|>,<|sid_start|><|sid_183|><|sid_461|><|sid_616|><|sid_768|><|sid_end|>`
> 
> <<User>>: Suggest a name and description for the bundle.
> 
> <<Model>>: Xbox Racing Legends: Speed & Style Pack














