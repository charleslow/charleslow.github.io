# Recent Trends in Search & Recommendations

This document summarizes some important ideas in both recommender systems and search systems that have emerged in the past few years. It aims to be a helpful read for someone trying to get acquainted with modern practices in these fields.

What we will cover:
- [Embedding learning](#embedding-learning). Embedding learning for items and users is an upstream task that are used for downstream retrieval and ranking.
- Recommendations
- Search retrieval and ranking

## Embedding Learning

Embedding learning is a foundational tool that powers all recommendation and search systems. Typically, users and items are represented by one or more embeddings that get fed into neural networks for prediction and recommendation tasks. 

Whilst many off-the-shelf LLM embedding models exist today, the best performing embeddings are often context-specific. For example, from the perspective of a job-seeker, we may want the embeddings for `Physical Education Teacher` and `Gym Trainer` to be similar, as the focus is on similar job functions / skills. However, from the perspective of a HR personnel trying to do manpower planning, we may want `Physical Education Teacher` to be more similar to `Executive at Singapore Sports School`, since the focus is more on eligible pathways for internal rotation (btw, I made this example up). Hence, there is a need to effectively learn embedding models in a specific product setting. Lee 2020 describes the task like so:
> (the goal is to) Learn embeddings that preserve the item-to-item similarities in a specific product context.

It is common practice to have an upstream pipeline to learn item embeddings that get re-used for multiple downstream tasks (let's call this a "universal" embedding). For example, YouTube uses a common set of video embeddings used by all its models [Lee 2020]. To adapt the embeddings to a specific task, the task model itself can always add a simple translation MLP layer to translate the universal embedding so that the embeddings can be more performant in each task setting.

Another way is to have each retrieval and ranker model learn its own embeddings for its specific task. 
- However, this leads to redundancy to store and update each set of embeddings, often in some in-memory feature store, which is costly. 
- This approach is also often not as performant as learning a universal set of embeddings and then adapting the embeddings to each task via a small learned MLP layer
- Training from a frozen set of universal embeddings is also much more <<compute efficient>>, as we do not need to forward/backward propagate into a potentially large language model that generates the embeddings from text or some other modality.

We can think of upstream training of an embedding model as analogous to the common practice of pre-training large language models in a semi-supervised manner on a large text corpus. The LLM is then fine-tuned for specific applications.

Although the performance of using multiple embeddings to represent an item has been proven to be more performant than using a single embedding (e.g. using ColBert), most companies seem to adopt a single embedding to reduce engineering complexity. For example, Pinterest briefly experimented with multiple embeddings per item but reverted to a single embedding to reduce costs.

There are 3 primary ways of embedding representation in the literature:
* <<ID-Based>>: This method maps an item's ID directly to an embedding.
    - $f(\text{item\_id}) \rightarrow \text{embedding}$
    - This is often used in a task-specific model, but not usually used for a universal embedding model shared by downstream tasks, as we cannot represent new items that have no user activity
* <<Content-Based>>: This is the most common approach, generating embeddings from an item's content, such as its text, images, or audio 
    - $f(\text{item\_text}, \text{item\_image}, ...) \rightarrow \text{embedding}$
    - Textual content is usually passed into an LLM to generate embeddings
    - For other modalities, task-agnostic preprocessing may be employed. For example, Lee 2020 sampling frames from a video and runs them through a ResNET model to create raw embeddings before they are fed into the main embedder model
* <<Graph-Based>>: This method can be seen as an extension of the content-based approach. It generates an embedding from the attributes of an item's neighbors in a network.
    - $f(\text{attributes of neighbours of item}) \rightarrow \text{embedding}$
    - This is particularly effective in social network products where network interactions are crucial, such as on LinkedIn, Pinterest, or Facebook
    - While powerful, this method is more computationally expensive at both training and inference times because it requires access to the graph to compute embeddings

Embeddings are typically trained in a <<contrastive learning>> manner, which simply means that we want to encourage related items (e.g. videos frequently co-watched) to have high similarity scores and unrelated items (e.g. randomly sampled vidoe pairs) to have low scores. 

Thus, we can think of contrastive learning as comprising 3 main components:
- <<Positive Sampling>>. How we mine for related items from data is a non-trivial, but an oft-neglected topic. Usually some behavioural statistics are used, e.g. Lee 2020 uses `videos frequently co-watched by users` to determine positive pairs. The guiding principle is to err on the side of being stricter in the selection of positive samples to minimize the appearance of false positives in the data (which are more harmful than false negatives).
    - In JumpStart, we have explored using Pointwise Mutual Information score to discard item pairs with low PMI score. Recall that the PMI score is computed as:
    $$
        pmi(item_a, item_b) = \log_2 \frac{p(item_a \text{ co-occuring with } item_b)}{p(item_a) \cdot p(item_b)}
    $$
    - Considering co-occurrence statistics focuses only on the numerator. Normalizing by the marginal probabilities of the items reduces the incidence of popular items being labelled as related when they are actually not
    - In my experience, the bottom `10%` of co-interacted items by pmi score can be safely discarded
    - Another subtle issue is that power users contribute disproportionately to positive pairs.
        - If a given user interacts with $n$ items, then he/she generates $\binom{n}{2} = \frac{n(n-1)}{2}$ item co-occurrence pairs
        - The quadratic scaling in $n$ means that the item interactions from power users may dominate that of all other users, but these are also the least informative item pairs :'( as power users may have less specific item preferences
        - Hence it is advisable to discard power users beyond a particular percentile to improve positive sampling 
- <<Negative Sampling>>. Much of the research literature focuses on this issue. The reason is that as research data (e.g. `Movielens`) is fixed, the positive pairs are usually pre-determined in the data. But the negative samples are usually not pre-specified in the data, so methods are devised to sample:
    - **Random sampling**. This is the simplest and most common approach: for a given anchor item, we sample negatives uniformly from the item catalogue, omitting the positive item(s) for the anchor
    - **Impressed but not positive**. This is also a simple approach if we have access to item impression data. The impressed items are often hard negatives because they were deemed to be relevant by the existing recommendation system and surfaced to the user.
        - The potential danger is that the model may only learn to distinguish negatives in a specific retrieval setting and is not robust to changes in the retrieval model
    - **Hard negative mining**. Various other methods exist to mine for hard negatives, e.g. using BM25 score or embedding similarity. We discuss two ideas on this below in a bit more detail.
    - Usually, it is best to mix negatives from all of the above so that the model learns to distinguish positives from a wide range of negatives 
- <<Loss>>. Again, much ink has been spilt on this topic, but there are really only two main losses that we need to know:
    - **Triplet loss**. 
    - **Cross entropy (or InfoNCE) loss**.

<<WORK IN PROGRESS>>

To improve training, two ideas are critical:
1.  **Semi-hard Negative Mining**: Originating from FaceNET, this technique addresses the problem of "false negatives" (items that are incorrectly labeled as negative) which can confuse the model. It focuses on training examples that are in a "goldilocks zone"—not too easy and not too hard. Lee (2020) demonstrated that mining for the hardest semi-hard negatives within a mini-batch is a computationally cheap and critical step for model performance.
2.  **Smart Negative Sampling**: Since random sampling often fails to find challenging negative examples, methods are devised to mine for them. This can involve using impression data, surrogate algorithms like BM25, or approximate-nearest-neighbor search. Lee (2020) also introduced using hierarchical clustering on a relational graph to find negatives from nearby clusters, which can be combined with semi-hard negative mining to prevent issues with false negatives.


### From Matrix Factorization to Sequential Recommendations

Traditional matrix factorization methods learn a single, static user representation to predict their entire interaction history. This is problematic for users with diverse or evolving interests, as the model learns a "diluted" representation. While time-decayed ratings can help by giving more weight to recent interactions, the model still fails to learn the sequential nature of those interactions fully.

Sequential recommenders address this by mirroring the actual recommendation task: given a user's history up to time *t*, predict their next action at time *t+1*. The user's representation is therefore dynamic and based on their recent actions. There are three main approaches to this:
* **Markov Approach**: A simple but often effective method that uses the last interacted item as the primary feature to predict the next one.
* **Pointwise Approach**: This method involves "rolling back" features to what was available at the time of each action in the training data. While this explicitly computes every user-item interaction, it requires significant data-loading effort and careful feature engineering. Google chose this approach for its simplicity in distributed model training.
* **Sequential Modelling Approach**: This is analogous to language modeling, where the model learns to predict the next item in a sequence, learning from many items simultaneously.

### Case Studies in Sequential Recommendations

#### **Covington (2016) - Pointwise Approach at YouTube:**
YouTube's influential 2016 paper detailed a pointwise approach for both retrieval and ranking.
* **Retrieval**: The model used features like the average embedding of the last 50 video watches and search queries, along with user features like gender. A small MLP was trained with a sampled softmax loss to predict the next video watch. A crucial finding was that predicting the *next* watch was far more effective in online A/B tests than predicting a randomly held-out watch from the user's history.

[Diagram: Pointwise Approach for Retrieval - Covington 2016 architecture from slide 13]

* **"Example Age" Feature**: To capture the time-sensitive nature of video popularity, an "Example Age" feature was introduced. This simple feature encodes when an action took place, allowing the model to learn the "popularity lifecycle" of items, where popularity often spikes shortly after upload and then fades.

[Diagram: Chart showing Class Probability vs. Days Since Upload with and without the "Example Age" feature, from slide 15]

* **Ranking**: The ranking model was similar to the retrieval model but incorporated many more features, particularly those related to user-item historical interactions (e.g., "how many videos from this channel has the user watched?"). The paper noted that even simple transformations of features, like square roots, improved performance, highlighting the reality of manual feature engineering.

[Diagram: Pointwise Approach for Ranking - Covington 2016 architecture from slide 16]

#### **SASRec (2018) - The Transformer-Based Approach:**
SASRec (Self-Attentive Sequential Recommendation) applied the transformer architecture to recommendations, treating a user's interaction history as a sequence of items to be predicted, much like words in a sentence.
* **Architecture**: Using causal masking, the transformer embedding at each time step encodes all previous interactions. The training loss is typically a binary cross-entropy loss comparing the dot product of the model's prediction with the target item embedding against a random negative sample. This approach significantly outperforms traditional matrix factorization methods.
* **Position Embeddings**: Since transformers have no inherent notion of sequence, SASRec adds a learned position embedding to each item in the input sequence. This helps the model weigh recent items more heavily, as shown by visualizations of the attention matrix.
* **Efficiency and Power**: This paradigm is more efficient than the pointwise approach because it learns from N time steps simultaneously for each user. It also reduces the need for manual feature engineering, as the model can implicitly learn user-item interaction features.

[Diagram: Visualization of Average Attention Matrix from SASRec, with and without Positional Embeddings, from slide 21]

A subsequent paper, **BERT4Rec (2019)**, proposed a "masked token prediction" task similar to BERT, allowing the model to use future items as input. While it initially appeared superior, Klenitskiy (2023) showed this was due to a difference in loss functions; when both use the same sampled softmax loss, **SASRec consistently performs better and trains faster**. The conclusion is that SASRec with sampled softmax loss is the current industry standard.

#### **PinnerFormer (2022) - Long-Term Sequential Modeling at Pinterest:**
PinnerFormer evolved from the need to switch from real-time user embedding computation to less costly daily batch jobs.
* **Long-Term Loss**: The key idea is that at each time step, instead of only predicting the very next item (like SASRec), the model predicts a random item from the user's future interactions over a longer window (e.g., 28 days). This creates more stable embeddings and surprisingly, PinnerFormer beats SASRec even when retrained in real-time.
* **Advanced Features**: PinnerFormer uses pre-computed PinSage (graph-based) embeddings as a base. It distinguishes between different action types (e.g., "Pin Save" vs. "Long Click") by concatenating a learned "action" embedding to the item embedding. It also heavily relies on time features, using Time2Vec to encode timestamps, noting a significant performance drop without them.
* **Training Details**: The model uses a combination of in-batch and random negatives with Log-Q correction, and ensures each user in a mini-batch is weighted equally to avoid bias from users with very long histories.

[Diagram: Illustration of PinnerFormer's "Long Term Loss" compared to SASRec from slide 25]

### Multi-Task Learning in Recommendations

Recommendation systems often need to optimize for multiple objectives (e.g., clicks, saves, purchases). Several architectures address this, primarily for the pointwise approach:
* **ESMM (2018)**: This model explicitly encodes the relationship that a conversion can only happen after a click (`P(convert) = P(click) * P(convert|click)`). This allows the conversion model to benefit from the abundant click data while learning from the sparse conversion data.

[Diagram: ESMM architecture from slide 29]

* **Shared-Bottom Architecture**: A common approach where bottom layers of a neural network are shared across tasks, with separate "towers" or heads for each specific task prediction. The final recommendation is a manually tuned weighted average of the logits from each head.

[Diagram: Shared-Bottom Model architecture from slide 30]

* **Mixture of Experts (MoE)**: An improvement over the shared-bottom model, MoE uses specialized "expert" layers. For each task, a gating network learns a weighted average over the outputs of these experts, allowing the model to modularize and handle potentially conflicting tasks more effectively.

[Diagram: Multi-gate Mixture-of-Expert (MoE) model architecture from slide 31]

* **PinnerFormer's Approach**: For sequential models, PinnerFormer took a simpler route, finding that treating all positive actions (repins, clicks, etc.) as equal signals was the best all-around strategy for their use case.

| Training Objective | 10s Closeup | 10s Click | Repin | All |
| :--- | :--- | :--- | :--- | :--- |
| **10s Closeup** | 0.27 | 0.02 | 0.09 | 0.17 |
| **10s Click** | 0.01 | 0.49 | 0.01 | 0.12 |
| **Repin** | 0.15 | 0.03 | 0.17 | 0.13 |
| **Multi-task** | 0.23 | 0.28 | 0.13 | 0.23 |

***

## II. Trends in Search Systems

While related to recommendations, search systems have distinct challenges and characteristics in practice.

### Search vs. Recommendation

* **Relevance is Stricter**: In search, there are objectively correct and incorrect results. Irrelevant results must not be surfaced, as they erode user trust.
* **Latency is Critical**: Users expect instant search results, making pre-computation of recommendations for a given query impossible.
* **LLMs Have a Larger Impact**: Search benefits more directly from advances in LLMs due to its text-centric nature and the overlap with fields like Information Retrieval and Question-Answering.

Search tasks exist on a spectrum from pure relevance (like academic Q&A) to pure personalization (which is essentially recommendation). E-commerce search lies in the middle, where relevance is key, but personalization plays a major role in user satisfaction given the large number of potentially relevant items for a broad query.

### Academic vs. Industry Search

Academic research often focuses on Question-Answering (Q&A) datasets like MSMarco or TriviaQA, where queries are well-formed questions with a single correct answer. In this setting, pre-trained LLMs perform exceptionally well, and it is easier to beat traditional baselines like BM25.

#### **Standard Academic Models:**
* **Ranking (Cross-Encoders)**: The standard is a cross-encoder architecture (often called MonoBERT), where a pre-trained language model like BERT takes a `[Query, Document]` pair as input and outputs a relevance score. Later papers showed that using a sampled softmax loss with many random negatives is more effective than the original binary cross-entropy loss. As models scale, so does performance, with **RankLlama** showing stronger results than older BERT or T5-based rankers. A 2023 paper by Sun even showed that zero-shot prompting with ChatGPT, using a "permutation generation" method and sliding windows, can achieve state-of-the-art performance on Q&A tasks without any fine-tuning.

[Diagram: Standard Cross Encoder / MonoBERT architecture from slide 39]

| Ranking Model | MRR@10 for MSMarco |
| :--- | :--- |
| monoBERT | 37.2 |
| RankT5 | 43.4 |
| LLaMA2-8b | 44.9 |

* **Retrieval (Two-Tower Models)**: The standard is Dense Passage Retrieval (DPR), which uses two separate encoders—one for the query and one for the document (passage). The model is trained with a sampled softmax loss to make the cosine similarity between a query and its positive document high, relative to negative documents. These negatives often include "in-batch" negatives (other positive documents in the same training batch) and "hard" negatives mined using BM25. To improve these retrievers, a common technique is **distillation**, where a powerful but slow cross-encoder (the "teacher") is used to train a faster two-tower model (the "student").

| Retrieval Model | MRR@10 for MSMarco |
| :--- | :--- |
| BM25 | 18.4 |
| ANCE (BERT-based) | 33.0 |
| LLaMA2-8b | 41.2 |

[Diagram: Illustration of a Mini Batch for Dense Passage Retrieval from slide 42]

### Complexities of Industry Search Systems

Industry search systems are significantly more complex, needing to balance relevance, latency, and personalization.

#### **Case Study: Pinterest Search (2024)**
* **Teacher-Student Distillation**: Pinterest fine-tunes a large teacher LLM (Llama-3-8B) on 300k human-labeled query-pin pairs. To improve the teacher's performance, the text representation for each pin is enriched with its title, description, AI-generated image captions, high-engagement historical queries, and common board titles it's saved to.

| Model | Accuracy | AUROC 3+/4+/5+ |
| :--- | :--- | :--- |
| SearchSAGE | 0.503 | 0.878/0.845/0.826 |
| mBERT_base | 0.535 | 0.887/0.864/0.861 |
| T5_base | 0.569 | 0.909/0.884/0.886 |
| mDeBERTaV3_base | 0.580 | 0.917/0.892/0.895 |
| XLM-ROBERTalarge | 0.588 | 0.919/0.897/0.900 |
| Llama-3-8B | 0.602 | 0.930/0.904/0.908 |

* **Scaling Up with Distilled Data**: This powerful teacher model then generates labels for 30 million un-labeled data points. A small, fast student MLP model is then trained on this massive distilled dataset. This student model recovers up to 91% of the teacher's accuracy and, because the teacher was multilingual, the student model also learns to handle multiple languages effectively despite being trained only on English human-annotated data.

| Training Data | Accuracy | AUROC 3+/4+/5+ |
| :--- | :--- | :--- |
| 0.3M human labels | 0.484 | 0.850/0.817/0.794 |
| 6M distilled labels | 0.535 | 0.897/0.850/0.841 |
| 12M distilled labels | 0.539 | 0.903/0.856/0.847 |
| 30M distilled labels | 0.548 | 0.908/0.860/0.850 |

[Diagram: Teacher-Student distillation process at Pinterest from slide 47]

#### **Case Study: Baidu Search (2021, 2023)**
* **Bootstrapping Labels with Weak Signals**: Instead of training a large teacher model, Baidu uses a simple decision tree model trained on "weak relevance signals" (e.g., click-through rates, long-click rates, skip rates) to predict a "calibrated relevance score" for unlabeled data. Training a cross-encoder on this calibrated score was shown to be far more effective than training on raw clicks alone.

[Diagram: Baidu's decision tree model over weak relevance signals from slide 48]

* **Query-Sensitive Summary**: To avoid the latency of processing full documents, Baidu developed an algorithm to extract the most relevant sentences from a document based on word-match scores with the query. This summary is then used as input for the ranker, significantly boosting offline performance.
* **Modular Attention for Speed**: To accelerate their cross-encoder, Baidu uses a modular attention mechanism. Self-attention is first applied independently within the `[query-title]` segment and the `document summary` segment for several layers, before a few final layers of full cross-attention are applied. This structure reduces computation and increases inference speed by 30%.

[Diagram: Baidu's Modular Attention architecture from slide 50]

* **From Relevance to Satisfaction**: A 2023 paper from Baidu argued that relevance alone is insufficient and that models must optimize for user satisfaction. They engineered features for quality (e.g., number of ads), authority (e.g., PageRank), and recency, and even built a query analyzer to determine if a query was authority or recency-sensitive. These features, along with relevance, were used to train a model that boosted performance by 2.5% over a pure relevance model. Notably, numeric satisfaction features were simply normalized and appended directly into the text input for the LLM to process.

#### **Case Study: Google DCNv2 (2020)**
Google's Deep & Cross Network V2 (DCNv2) addresses a weakness in standard MLP rankers: they are not ideal for explicitly modeling feature interactions (e.g., `user_is_interested_in_topic_X` AND `item_is_about_topic_X`). DCNv2 introduces explicit "cross layers" that are designed to generate these feature crosses efficiently before feeding them into a standard deep network. This architecture provides performance gains over a standard MLP with the same number of parameters.

[Diagram: Google DCNv2 architecture with Cross Layers from slide 53]

#### **Case Studies: Embedding-Based Retrieval (EBR) at Facebook and Taobao**
EBR systems are used for semantic search but can struggle with relevance compared to traditional keyword-based lexical matching.
* **Facebook's Hybrid Approach**: Facebook integrates nearest-neighbor search directly into its in-house lexical search engine. A query can contain both standard lexical terms (e.g., `location:seattle`) and an `nn` operator that finds items within a certain embedding-distance radius. This ensures documents fulfill both lexical and semantic requirements. Facebook's query and document encoder towers use a mix of term-based (char-3-grams) and LLM-based representations.
* **Taobao's Relevance Filters**: Taobao found their EBR system sometimes returned irrelevant but semantically close items (e.g., a search for "Nike shoes" returning "Adidas shoes"). Their solution was to apply explicit lexical boolean filters on top of the ANN search (e.g., `ANN search + Brand:Nike + Product:Shoes`). While this sacrifices some of the "fuzzy" semantic search capability, it guarantees relevance. Taobao also found that optimizing for engagement (clicks) does not guarantee relevance, and ultimately chose to accept slightly lower engagement to ensure a higher percentage of relevant results. Their query representation is enhanced by using cross-attention between the query and the user's historical interactions to personalize the query embedding.

| Experiment | Engagement % | Relevance % |
| :--- | :--- | :--- |
| Baseline | 85.6% | 71.2% |
| Baseline + personalization | 86.4% | 71.4% |
| Baseline + Lower temperature | 85.5% | 79.0% |
| Baseline + all | 84.7% | 80.0% |