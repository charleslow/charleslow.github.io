Summaries of individual papers that I have taken a deeper look into.

# Quick Reads

Here I put down quick summaries of papers I read quickly. May possible revisit these with its own page if a deep dive is warranted.

## JEPA

### [Assran 2023 - Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)

This is the I-JEPA paper. This paper aims to perform JEPA-style self-supervised learning for vision transformers without using image augmentations (like rotation, shear etc.). The main idea is to take a random image patch as <<context>> and aim to predict multiple other image patches as <<target>>. Importantly:
- The loss is to minimize l2-distance between the $\text{Predictor}(\text{Context Encoding})$ and $\text{Target Encoding}$ in latent space
- To avoid representation collapse in latent space, the target encoder is updated as a trailing exponential moving average of the Context Encoder
- This approach is simple and scalable and matches performance of SOTA methods using image augmentations

### [Geng 2022 - Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)](https://arxiv.org/abs/2203.13366)

This is one of the earliest papers to reframe recommendation task as a language modelling task. The idea is to represent user-item interaction data and other metadata in recommendation tasks as an `input` to `response` task for supervised fine-tuning (or pre-training). Then we can have a unified language model to perform various tasks in a recommendation system.

The tasks proposed are:
- <<Sequential Recommendation>>: Given purchase history `item_123, item_456`, what is the next item? `1581`
- <<Rating Prediction>>: What star rating would `user_123` give to `item_456`? 5.0
- <<Explanation Generation>>: What explanation or review would `user_123` give to `item_456` (iphone case)? You can use it to protect your phone
- <<Review Summarization>>: Give a short sentence summarizing this review `<insert review>`. `<insert summary>`
- <<Direct recommedation>>: Pick the most suitable item from the following list to recommend to `user_123`: `item_123, item_456, ...`. `item123`

These tasks are mixed together and then used to pretrain an encoder-decoder model from scratch, and then used for various recommendation tasks.

### [Tang 2023 - SE-DSI](https://arxiv.org/abs/2305.15115)

This is an expansion of the [Differentiable Search Index](./papers/tay_2022.md) paper. The argument is that teaching an LLM to learn arbitrary document ids is not semantic. Hence we should represent documents as semantic words to aid learning.

Specifically:
- <<Docid Representation>>. For each document, use a T5 model to generate a query to serve as document identifier (5% collision).
- <<Input features>>. Use 3 types of input features to predict the "query docid":
    - Whole document
    - Key passages identified using textrank
    - Key sentenes identified using textrank

This approach beats vanilla DSI, as we use semantic information as document representation. Note that evaluation task is Q&A.

### [Chen 2023 - Understanding DSI for Text Retrieval](https://arxiv.org/abs/2305.02073)

Analyzes the shortcomings of DSI through 3 metrics:
- <<Exclusivity>>: There should be a one-to-one mapping between document contents and semantic ID.
    - `Task`: Given first N tokens of document, can we retrieve the correct ID
- <<Completeness>>: The LLM should remember as much information about each document as possible.
    - `Task`: Use BERT model to find important chunks in document, then use these chunks as query to see if we can retrieve the correct ID.
- <<Relevance>>: Can the LLM rank documents in relevance order to a query.
    - `Task`: For a given query, check the document at position `k` of the LLM's recommendation order to see how relevant it is. Specifically, measure the cross encoder score of document $d_k$ against the query and compare to score of a random document $d_r$. Lower score is considered a failure
    - Found that DSI only generates relevant documents at position `1`, and then becomes highly irrelevant thereafter.

Proposed improvements:
- <<Data Filtering>>: Break each document into chunks and use a teacher model `TCT-Colbert` to perform retrieval using that chunk as query. Only successful retrievals that get the document back are considered useful content used for LLM fine-tuning.
- <<Multi-task learning>>: Normal DSI indexing task only seeks to predict one document ID given its contents or a query. They propose to add an additional task where we get LLM to learn to predict a list of relevant document IDs.
    - The list of relevant document IDs are generated using a teacher model (`TCT-Colbert` again).

### [Ju 2025 - GRID](https://arxiv.org/abs/2507.22224)

Snap's GRID repository which is a modular repo for semantic ID based generative retrieval to facilitate architecture comparison. A generative retrieval system has the following components:
- <<Semantic ID Tokenization>>:
     - Convert item description to embedding
     - Quantize embedding to tokens (K-means, VQ-VAE, RQ-VAE)
- <<Sequential Recommender>>:
    - Whether to use user token
    - Data augmentation using sliding windows
    - Architecture: Encoder-decoder vs decoder only
    - Beam search

<<Note:>> this system only considers the case where each user is represented as a sequence of items, and the LLM is trained from scratch (not finetuned from pretrained).

Their experiments are interesting:
- <<Residual K-means tokenization is best>>, compared to VQ-VAE and RQ-VAE. This is quite surprising and I'm not 100% convinced.
- <<Larger OTS embedding models are better>>, no surprise.
- <<Encoder-decoder sequential recommender is clearly better>>. Somewhat surprising, I guess there is some inductive bias that is useful in this setting to have full attention over the user sequence.
- <<Sliding window is better>>. Sliding window means that given a training data sequence of a user represented by item interactions, we generate all possible contiguous sequences to augment our dataset. Not surprising.
- <<SID de-duplication is unnecessary>>. Between appending an arbitrary token to deduplicate SIDs or just randomly selecting from SIDs with multiple items hashed to it, the performance difference is negligible. This is assuring that we don't need to further complicate the system.
- <<Constrained beam search is unnecessary>>. Constrained beam search means that we maintain a trie to only generate tokens leading to a valid SID. The results show that this has negligible performance difference to unconstrained beam search, which has much better efficiency. This is assuring that we can just use uncontrained beam search.

