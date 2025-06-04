# Solatorio 2024 - GISTEmbed

[GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning](https://arxiv.org/abs/2402.16829)

This paper proposes a way to learn embeddings using contrastive learning. The main idea is to use a guide model to filter out false negatives from training for better data quality and performance. GISTEmbed models are currently topping the MTEB benchmark.

## Implementation

The method has an [implementation](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/GISTEmbedLoss.py) in the SentenceTransformer package. We will walk through the implementation here.

Firstly, the loss is initialized with both a `model` to train and a `guide` model, both of which are `SentenceTransformers`.

In the `forward` step:
- `sentence_features` is the input which is a list of `dict[str, Tensor]`
    - The list is length `2` if we have anchor, positive
    - The list is length `3` if we have anchor, positive, negative
- The `sentence_features` is passed into both the `model` and the `guide` to get the respective embeddings
    - For `guide`, we may need to re-tokenize it if the tokenizer of the `guide` differs from `model`
    - This is done by using `batch_decode` and then `tokenize` again
- Now we have `anchor`, `positive` and `negative` embeddings, each of shape `(batch_size, embed_dim)`
- The `sim_matrix` is used to compute pairwise cosine similarities:
    - The code is simply `torch.nn.CosineSimilarity(dim=-1)(embed1.unsqueeze(1), embed2.unsqueeze(0))`
    - `embed1` becomes shape `(batch_size, 1, embed_dim)` and embed2 becomes shape `(1, batch_size, embed_dim)`
    - The similarity is compared at dimension `-1`
    - Broadcasting ensures that the comparison is done pairwise, such that the result is of shape `(batch_size, batch_size)`
    - This is a common way to do pairwise similarity
- Now we obtain the pairwise similarity matrices:
    - `ap_sim`, `aa_sim`, `pp_sim`, `an_sim`
    - `guided_ap_sim`, `guided_aa_sim`, `guided_pp_sim`, `guided_an_sim`
- The anchor positive similarity threshold is used to filter away potential false negatives
    - This is simply the `guided_ap_sim.diagonal()` which corresponds to the similarity between the anchor and positive in each row
    - Note that they use the guide model for determining the threshold
    - This threshold is called `guided_sim`
- `mask_false_negatives` is used to suppress false negatives
    - Using the `absolute` strategy, cases where the `guided_sim_mat > guided_sim - self.margin` will be suppressed (set to `torch.inf`)
    - The idea is that negatives should not have a higher similarity than the threshold, otherwise there is a higher probability they are false negatives
    - This function is applied to `ap_sim`, `aa_sim`, `pp_sim` and `an_sim` to mask false negatives
- Finally, we have `scores`
    - `scores = torch.cat([ap_sim, aa_sim, pp_sim, an_sim], dim=1) / self.temperature`
    - This is of shape `(batch_size, 4*batch_size)`
- We create `labels` which is `torch.arange(batch_size)`
    - This is because the correct label is in the diagonal of `scores` matrix
- Finally the loss is computed via `torch.nn.CrossEntropyLoss()(scores, labels)`
    - Each row is considered as a classification task where the label corresponds to the column position where the correct class is found
    - The log softmax loss is then computed per row and averaged across rows
