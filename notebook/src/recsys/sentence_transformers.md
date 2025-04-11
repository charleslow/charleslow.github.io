# SentenceTransformers

SentenceTransformers is a useful library for training various BERT-based models, including two-tower embedding models and cross encoder reranking models.

## Cross Encoder

SentenceTransformers v4.0 updated their cross encoder training interface (see [the v4.0 blogpost](https://huggingface.co/blog/train-reranker)). Here we try to follow the key components for cross encoder training using their API. 

The main class for training is `CrossEncoderTrainer`. We rely on a Huggingface `datasets.Dataset` class to provide training and validation data. `CrossEncoderTrainer` requires that the dataset format matches the chosen loss function.

The [loss overview page](https://sbert.net/docs/cross_encoder/loss_overview.html) provides a summary of cross encoder losses and the required dataset format. In general for cross encoder training, we have two sentences which are either positively or negatively related to each other. Which loss function we choose depends on the specific dataset format we possess.

### BinaryCrossEntropyLoss

Use this loss if we have inputs in the form of `(sentence_A, sentence_B)` and a label of either `0: negative, 1: positive` or a float score between `0-1`. In the huggingface `dataset`, we would need to ensure that the label column is named `label` or `score`, and have two other input columns corresponding to `sentence_A` and `sentence_B`. For `sentence_transformers` package in general, order of columns matter, so we should set it to `sentence_A, sentence_B, label`.

Inspecting the [source code](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/losses/BinaryCrossEntropyLoss.py#L9-L116) would show that each sentence pair is tokenized and encoded by the cross encoder model. The cross encoder must output a single logit (i.e. initialized with `num_labels=1`). Thus we get a prediction vector $x$ of dim `batch_size`. The `torch.nn.BCEWithLogitsLoss` is then used to compute the binary cross entropy loss of the prediction logits against the actual labels $y$, according to the standard bce loss:
$$
    L(x_i, y_i) = -w_i \left[
        y_i \cdot \log \sigma(x_i) + (1-y_i) \cdot \log(1 - \sigma(x_i))
    \right]
$$

This is a simple and effective loss. The user should ensure that the labels are well distributed (between $0$ and $1$) without any severe class imbalance. 

### CrossEntropyLoss

The <<CrossEntropyLoss>> is used for a classification task, where for a given input sentence pair `(sentence_A, sentence_B)`, the label is a class. For example, we may have data where each sentence pair is tagged to a `1-5` rating scale. We need to instatiate the `CrossEncoder` class with `num_labels=num_classes` for this use case. This creates a prediction head for each class.

Looking at the [source code](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/losses/CrossEntropyLoss.py#L8-L84), we see that this loss simply takes the prediction logits from the model (of dimension `num_labels`) and computes the `torch.nn.CrossEntropyLoss` against the actual labels. 

Note that the cross entropy loss takes the following form. Given `num_labels=C` and logits of $x_1, ..., x_C$, where the correct label is index $y$, we have:

$$
    L(x, y) = - \log \left(
        \frac{e^{x_y}}{\sum_{j=1}^C e^{x_j}}
    \right)
$$

### MultipleNegativesRankingLoss

This is basically InfoNCE loss or in-batch negatives loss. The inputs to this loss can take the following forms:
- `(anchor, positive)` sentences
- `(anchor, positive, negative)` sentences
- `(anchor, positive, negative_1, ..., negative_n)` sentences

The [documentation page](https://sbert.net/docs/package_reference/cross_encoder/losses.html#multiplenegativesrankingloss) has a nice description of what this loss does: Given an anchor, assign the highest similarity to the corresponding positive document out of every single positive and negative document in the batch.

Diving into the [source code](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/losses/MultipleNegativesRankingLoss.py#L12-L191):
- The inputs are `list[list[str]]`, where the outer list corresponds to `[anchor, positive, *negatives]`. The inner list corresponds to the batch size.
- `scores` of dimension `(batch_size)` are computed for each anchor, positive pair
- `get_in_batch_negatives` is then called to mine negatives for each anchor. 
    - candidates (positive and negatives) are extracted at `inputs[1:]` and flattened into a long list
    - A mask is created such that for each anchor, all the matching positive and negative candidates are masked out (not participating)
    - The matching negatives do not participate because they will be added later on
    - Amongst the remaining negatives, `torch.multinomial` is used to select `self.num_negatives` number of documents per anchor at random
    - `self.num_negatives` defaults to `4`
    - These randomly selected negative texts are then returned as `list[str]`
- For each negative in num_negatives mined in-batch negatives:
    - `score` of dimension `(batch_size)` is computed for the anchor, negative pair
    - The result is appended to `scores`
- Similarly, for each hard matching negative:
    - `score` of dimension `(batch_size)` is computed for the anchor, hard negative pair
    - The result is appended to `scores`

Now `scores` is passed into `calculate_loss`:
- Recall that `scores` is a list of tensors where the outer list is size `1 + num_rand_negatives + num_hard_negatives`, and each tensor is of dimension `batch_size`
- Thus `torch.cat` + `tranpose` is called to make it `(batch_size, 1 + num_rand_negatives + num_hard_negatives)`
- Note that for each row, the first column corresponds to the positive document
- Hence the `labels` may be created as `torch.zeros(batch_size)`
- Then `torch.nn.CrossEntropyLoss()(scores, labels)` may be called to get the loss

This sums up the loss computation for `MultipleNegativesRankingLoss`.

### CachedMultipleNegativesRankingLoss