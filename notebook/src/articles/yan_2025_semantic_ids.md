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
    - Tried $\beta = 0.25, 0.5, 1.0$. $\beta$ is the commitment weight that balances reconstruction accuracy with codebook commitment
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

## Code

Here we deep dive into Eugene's code and how it is implemented. Most of the code is contained in the [`/src` directory](https://github.com/eugeneyan/semantic-ids-llm/tree/main/src).

- [device_manager.py](#device_managerpy)
- [tokenize_items.py](#tokenize_itemspy)
- [embed_items.py](#embed_itemspy)
- [train_rqvae.py](#train_rqvaepy)
    - [QuantizationOutput](#quantizationoutput)
    - [VectorQuantizer](#vectorquantizer)
    - [RQVAE](#rqvae)
- [finetune_qwen3_8b_vocab.py](#finetune_qwen3_8b_vocabpy)
    - [FineTuneConfig](#finetuneconfig)
    - [extend_tokenizer](#extend_tokenizer)
    - [prepare_model](#prepare_model)
    - [load_sid_dataset](#load_sid_dataset)
    - [DataInspectionCallback](#datainspectioncallback)
    - [EmbeddingMonitorCallback](#embeddingmonitorcallback)
    - [SemanticIDGenerationCallback](#semanticidgenerationcallback)
    - [train_embeddings](#train_embeddings)


### device_manager.py

The `DeviceManager` detects the device (`cpu` or `cuda` or `mps`) and is instantiated early in the scripts. The interesting part `torch.set_float32_matmul_precision("high")` performed when device is `cuda`. It's supposed to speed up `float32` operations?

### tokenize_items.py

This script tokenizes the product descriptions of video games:
- Uses `Qwen/Qwen3-Embedding-0.6B`
- Batch size = `32`, max length = `2048`
- Reads the data from `data/output/Video_Games_items_updated.parquet` into polars
- Looks for the `item_context` field (already preprocessed)
- Uses the following prompt which will be tokenized
```
Instruct: Given a product description, generate a semantic embedding that captures
    its key features and characteristics.
Query: {original_item_text}
```
- Saves tokenized `input_ids` and `attention_masks` and saves them in `.npz` (compressed numpy) format using `np.savez_compressed`

### embed_items.py

Takes the tokenized file and embeds them.
- Writes embedded items into a parquet file

### train_rqvae.py

Compresses an item embedding into hierarchical semantic IDs.

The `RQVAEConfig` defines the following parameters:
- `item_embedding_dim`: embedding dim of our embedding model
- `encoder_hidden_dims`: `[512, 256, 128]` the size of the VAE encoder
- `codebook_embedding_dim`: `32` Dimension of codebook vectors
    - <<Qn:>> this does not need to match qwen embedding dim?
- `codebook_quantization_levels`: `3` levels in the codebook
- `codebook_size`: `256` number of codes per level
- `commitment_weight`: `0.25` Commitment loss weight (beta)
- `use_rotation_trick`: `True` Use rotation trick for better gradient flow
- `batch_size`: `32768` training batch size (why so large?)
- `gradient_accumulation_steps`: `1`
- `num_epochs`: `20000`
- `scheduler_type`: `cosine_with_warmup`
- `warmup_start_lr`: `1e-8` used for cosine_with_warmup
- `warmup_steps`: `200` used for cosine_with_warmup
- `max_lr`: `3e-4` maximum learning rate (start of cosine)
- `min_lr`: `1e-6` minimum learning rate (end of cosine)
- `use_gradient_clipping`: `True`
- `gradient_clip_norm`: `1.0`
- `use_kmeans_init`: `True` initializes codebook vectors using k-means
- `reset_unused_codes`: `True` reset unused codes periodically to avoid collapse
- `steps_per_codebook_reset`: `2` Reset unused codebook codes every N steps
- `codebook_usage_threshold`: `1.0` only reset if usage falls below this proportion (0-1)
- `val_split`: `0.05`
- `steps_per_train_log`: `10` log every N steps
- `steps_per_val_log`: `200` validate and checkpoint every N steps

`EmbeddingDataset` is a torch dataset holding the embeddings:
- Extracts all embeddings and holds them in a `torch.tensor` at init
    - Not worried about OOM?

#### QuantizationOutput

`QuantizationOutput` is used to hold data for one quantized item:
- Holds the local loss for one codebook layer
- Subclasses `NamedTuple` which is more lightweight than `dataclass`
- `quantized_st: Tensor`
    - The quantized vector which is passed onto the next layer
    - Has the "gradient trick" (either straight through or rotation trick) applied
    - Allows backpropagation into the encoder even though we passed through the non-differentiable codebook layer
- `quantized: Tensor`
    - The raw nearest neighbour vectors from the codebook
    - No gradients
    - <<Observation>> `quantized` and `quantized_st` should be identical in values, just that one has gradients attached
- `indices: Tensor`
    - These are integer indices which represent the semantic IDs
- `loss: Tensor`
    - The combined loss for this specific codebook layer
    - `loss = codebook_loss + beta * commitment_loss`
- `codebook_loss: Tensor`
    - Measures how well the codebook vector matches the encoder output
- `commitment_loss: Tensor`
    - Measures how well the encoder output matches the codebook vectors

#### VectorQuantizer

The `VectorQuantizer` implements one layer of the codebook and is the meat of the logic for training RQVAE. It will be stacked together layer multiple times to form the codebook.
- At initialization:
    - Initializes with `RQVAEConfig` to hold parameters
    - Initialize `self.embedding` to an embedding of size `codebook_size=256, codebook_embedding_dim=32`
        - Uniform initialization `self.embedding.weight.data.uniform_(-1 / self.codebook_size, 1 / self.codebook_size)`
    - Registers some buffers for tracking codebook usage:
        - `self.register_buffer("usage_count", torch.zeros(self.codebook_size))`
        - `self.register_buffer("update_count", torch.tensor(0))`
- `find_nearest_codes(x)`:
    - Takes an input vector `x`, compares it to all vectors in the codebook, and returns the nearest one
    - Simply uses `torch.cdist` to compute distances, then `torch.argmin` to get the nearest
    - Returns a tuple of torch tensors:
        - The nearest index (i.e. codeword) to `x`
        - The quantized embedding at the index position
- `forward(x)` -> `QuantizationOutput`:
    - Finds the nearest index and quantized embeddings for a batch of `x` 
        - Call `find_nearest_codes` to get `indices` and `quantized`
    - Applies the gradient estimator to get `quantized_st`
        - This will be used for gradient backprop to the encoder later
        - `apply_gradient_estimator` either uses the straight through or rotation method
    - Compute losses:
        - `codebook_loss` is the MSE loss between `x.detach()` and `quantized`
            - We want to pull the codebook embeddings toward `x`
        - `commitment_loss` is the MSE loss between `x` and `quantized.detach()`
            - We want to pull encoder output toward codebook embeddings
        - `loss = codebook_loss + beta * commitment_loss`
    - Everything is packaged into `QuantizationOutput` and returned
    - `self.update_usage` is also called:
        - Updates counts of which indices were the nearest to `x`
        - Updates the number of training steps
- Straight through
    - The straight through gradient estimator simply returns `x + (quantized - x).detach()`
    - Essentially, the embeddings passed forward is `quantized`
    - But the vector used for gradient backprop is `x` (hence straight-through back to the encoded `x`)
    - This is a naive method but works well enough
- Rotation
    - The problem with the straight through estimator is that we use `quantized` for the forward pass but use `x` for the backward pass
        - This can be problematic especially if `q` and `x` are far apart
    - The rotation idea is to apply a rotation to `x` until it aligns with `q`
        - Since the rotation is differentiable, we get better gradients back to `x`
    - We compute (to check later):
        - Let $u = x / ||x||$
        - $w = \frac{u + q}{||u + q||}$ is the halfway vector between `u` and `q`
        - $x_{rot} = x - 2 \langle x, w \rangle w + 2 \langle x, u \rangle q$
- `reset_unused_codes`
    - Look up `self.usage_count` to find unused indices (used `0` times)
    - Take the current batch of encoded data, and randomly select them to become the new codebook vectors
    - This makes it likely for them to be used in the next forward pass since they correspond to actual encoder outputs 
    - All usage counters are reset after this

#### RQVAE

The RQVAE class now assembles multiple `VectorQuantizer` into the actual VAE to create semantic IDs.

At initialization:
- `self.encoder`: a simple MLP that shrinks the input embedding down to the codebook dimension
    - In this code, we go from `1024 -> 512 -> 256 -> 128 -> 32`
- `self.decoder`: a simple MLP that goes backward from quantized vector up to embedding dimension
    - In this code, the decoder dims are just the reversed of the encoder dims
    - So we go from `32 -> 128 -> 256 -> 512 -> 1024`
- Both encoder and decoder are wrapped in `nn.Sequential`
- `self.vq_layers` contains the `VectorQuantizer`s
    - It is an `nn.ModuleList` of `3` `VectorQuantizer`s
- `forward`: the main magic of this class
    - First we encode input item embedding `z = self.encode(x)`
    - Also init `residual = z`
    - Init `quantized_out = torch.zeros_like(z)`
        - The quantization output will be the sum of 
    - Now we run a for loop through the vector quantizer layers:
        - First we compute the quantization output for this level (which contains the mapped ID for this level etc.)
            - `vq_output: QuantizationOutput = vq_layer(residual)`
        - Then we update the residual by subtracting the nearest codebook vector
            - `residual -= vq_output.quantized.detach()`
        - We accumulate the codebook vectors (with gradients) into `quantized_out`
            - `quantized_out += vq_output.quantized_st`
            - Recall that the final representation passed to the decoder is $\hat{z} = \sum_{l=1}^L e_l$
        - We also accumulate the loss for each layer
            - `vq_loss += vq_output.loss`
            - This is the codebook + commitment loss, reconstruction loss comes later
    - Finally we get the total loss
        - Compute the reconstruction loss
            - `x_recon = self.decode(quantized_out)`
            - `recon_loss = F.mse_loss(x_recon, x)`
            - `loss = recon_loss + vq_loss`
- `encode_to_semantic_ids`: encodes an item embedding `x` to an integer tensor representing its semantic ID
- `decode_from_semantic_ids`: decodes an integer tensor `semantic_ids` by looking up the codebook, summing up the levels and passing back into the `decoder`
- `kmeans_init`
    - Runs kmeans on one batch of embeddings to initialize the codebook vectors
    - Runs kmeans to get `256` centroid vectors
    - Copies these vectors into the codebook directly
    - Process layer by layer

### finetune_qwen3_8b_vocab.py

This script performs <<Stage 1>> of the qwen fine-tuning. It focuses on extending the vocabulary to include new semantic ID tokens and trains embeddings for these new tokens.

#### FineTuneConfig

Dataclass containing config for the training
- `model_name`: `unsloth/Qwen3-8B`
    - <<Qn:>> Not instruction fine tuned?
- `load_in_4bit`: Set to `False` for embedding training
- `load_in_8bit`: Set to `False`
- `num_proc`: `32`
- `enable_thinking`: `False` we don't need thinking mode
- `extend_vocabulary`: `True`
- `codebook_levels`: `4`
- `codebook_size`: `256`
- `num_semantic_tokens`: `1024`
- `system_prompt`: see below
- `max_training_samples`: `32000` limit for training embedding
- `learning_rate`: `1e-3`
- `batch_size`: `32`
- `max_steps`: `1000`

The system prompt is as follows:
> "You are a helpful AI assistant that understands and works with semantic IDs for product recommendations. Semantic IDs are hierarchical identifiers in the format `<|sid_start|><|sid_105|><|sid_307|><|sid_705|><|sid_769|><|sid_end|>` that encode product relationships and categories. /no_think"

#### extend_tokenizer

`extend_tokenizer(model, tokenizer, config: FineTuneConfig)` adds semantic ID tokens to the tokenizer using Unsloth's `add_new_tokens`.
- Note that the vocab size affects two places:
    - `model.get_input_embeddings().weight`: the input embeddings
    - `model.get_output_embeddings().weight`: the language model head which predicts the next token
- First, we make sure that the vocab size of the tokenizer matches the vocab size of both the input and output embeddings
    - We need to call `model.resize_token_embeddings` to get the model embedding sizes to match the `tokenizer`
    - This is because the model embeddings are padded to be multiples of `128` for CUDA optimization reasons
- Next, we add new tokens using `unsloth.add_new_tokens`:
    - Special tokens of `<|rec|>`, `<|sid_start|>`, `<|sid_end|>`
    - Semantic IDs of `<|sid_0|>` to `<|sid_1023|>`

#### prepare_model

Prepares the model for training with some additional checks:
- Freezes gradients for all parameters
- Unfreezes only the weights for the `model.get_input_embeddings()` and `model.get_output_embeddings()`
- Checks the trainable parameter %

#### load_sid_dataset

Loads the semantic IDs training dataset:
- Checks if there are texts like `<|sid_start|>` to make sure processing is correct
- Applies chat template to the rows (but keeps as text)

There are 5 distinct categories of training data:
- <<SemanticID -> text>>:
    - **Input**: "Product `<|sid_start|>...<|sid_end|>` has title:"
    - **Output**: "Super Mario Bros" 
    - **Variations**: ID to title, description, category, features or full context
- <<Text -> SemanticID>>:
    - **Input**: "The product Super Mario Bros has SemanticID:"
    - **Output**: "`<|sid_start|>...<|sid_end|>`"
    - **Variations**: Similar variations to above
- <<Sequential Recommendation>>:
    - **Input**: "Based on recent purchases etc., next item:"
    - **Output**: "`<|sid_start|>...<|sid_end|>`"
    - **Variations**: Various sequence lengths of 2, 3, or 5 items.
- <<Semantic Understanding>>:
    - **Input**: "Products starting with `<|sid_start|><|sid_64|> are typically:`
    - **Output**: "Nintendo switch games"
    - **Variations**: Prefix to category, prefix to examples, similar items.
- <<Multi-hop Reasoning>>:
    - **Input**: "A user who bought `<|sid_a|>` might also buy:"
    - **Output**: "`<|sid_b|>`
    - **Variations**: Co-purchase patterns.
#### DataInspectionCallback

Used to inspect training data and tokenization at each training step, by simply logging them to console.

Patterns:
- `DataInspectionCallback` subclasses `transformers.TrainerCallback`
- `on_train_begin(self, args, state, control, **kwargs)`:
    - Checks the first batch of `train_dataloader`
    - Checks batch keys
    - Check shape of `batch['input_ids']`
    - Check shape of `batch['attention_mask']`
    - Check tokens and decoded of first row etc.
- `on_log(self, args, state, control, logs=None, **kwargs)`:
    - Only runs if `state.global_step % args.logging_steps == 0`
    - Check number of SID tokens
    - Decode first example and check

#### EmbeddingMonitorCallback

This callback aims to check how our embeddings are shifting over time.
- At initialization (or `on_train_begin`), we copy the state of the initial embeddings and clone detach them
- At each step, we compute the mean of the absolute difference between the current embeddings and the initial or state of embeddings from the previous step
- We also compute the per level codebook vector means etc.
- These are all logged to `wandb`

#### SemanticIDGenerationCallback

This is a qualitative check to answer the question "If I ask the model for a recommendation right now, does it use the semantic ID tokens or does it just output plain text?"
- A fixed set of test cases are used
- The test cases are `apply_chat_template`, then passed into the `tokenizer`, then `model.generate` and `model.decode`
- The messages are checked whether successful (SemanticIDs generated) and success rate is tracked
- Actual completion examples are logged into wandb as well

#### train_embeddings

The main method. Essentially we are just using unsloth `SFTTrainer` to do the training.

First, we set up `trl.SFTConfig` with a lot of the configuration we previously defined.
- Note that `dataset_text_field="text"`
- `report_to="wandb"`

The trainer `trl.SFTTrainer` is initialized with the model, tokenizer, datasets, config and callbacks.

Then, `trainer.train()` is called. 

Note that the model and tokenizer are initialized using `unsloth.FastLanguageModel` to use unsloth's optimized triton kernels.

### finetune_qwen3_8b_full.py

The code is structurally very similar to the vocab finetuning run. The difference is that we are doing full training, so we unfreeze all parameters. Consequently, the learning rate needs to be much lower at `2e-5`.
- Load the model from stage 1, namely `models/qwen3_8b_vocab_extended/final`
- A lot of the script focuses on the callbacks to evaluate recommendation quality












