# Yan 2025 - LLM for Recsys

[Recsys Keynote: Improving Recommendation Systems & Search in the Age of LLMs - Eugene Yan, Amazon](https://www.youtube.com/watch?v=2vlCqD6igVA)

This talk covers future of recsys and how LLMs can be incorporated. 3 challenges:
- Cold start challenge of hash based item IDs
- Lack of metadata
- Task specific models duplicate engineering, increase maintenance cost and don't benefit from transfer learning
    - Benefits: simplifies systems, reduces maintenance and transfer learning
    - But there may be alignment tax

## Kuaishou Example for semantic IDs

- Challenge: hash based item IDs don't encode item content and struggle with cold start and sparsity problem.
- Solution: Semantic IDs based on multimodal content

Kuaishou is a short video platform. The main problem they wanted to tackle is to help users discover new items faster.

Idea:
- Train standard ID-based embeddings for user and items
- Create cluster ID from concatenated content embeddings
    - Text: BERT
    - Video: ResNet
    - Audio: VGGish
- Run k-means on `100 million` items into around `1k` clusters
    - Each cluster gets an ID and also an embedding
    - Incorporate cluster embedding in final embedding

Result:
- `+3.4% clicks`
- `+3.0% likes`
- `+3.6% cold start coverage` (% of item impressions which are new items)
- `+1.2% cold start velocity` (% of new items that were able to hit some threshold of views)

- Example:
    - trainable, multimodal, semantic IDs @ Kuaishou
    - Short videos platform
    - Problem: help users discover new items faster

## Filtering Bad job recommendations at Indeed

Problem: poor user experience of email job recommendations and lost trust due to low quality job recommendations
Solution: Lightweight classifier trained from GPT-4o annotated data to filter bad recs

Process:
- Start with evals - 250 labelled examples with confidence labels
- Started with open LLMs like Llama2, but performance was very bad
- Used GPT-4, which was very accurate but too slow and costly (22 secs)
- Used GPT-3.5, but had poor precision (0.63) on job recommendations
- Finetuned GPT-3.5 and got 0.9 precision at 1/4th of of GPT-4 cost and latency
- Distilled lightweight classifier on finetuned GPT-3.5 labels

Lightweight classifier was `0.86 auc-roc`, with latency `<200ms`. Result was:
- `-18% bad recommendations`
    - Expected lower application rates because recommending fewer items
- `unsubscribed rate -5%`
- `application rate +5%`

## Enriching exploratory search queries @ Spotify

Problem: Help users search for new items (podcasts, audiobooks) in a catalogue of known items (e.g. songs, artists)
- How to solve cold start issue for new categories?
- Exploratory search was essential to expand beyond music

Solution: Query recommendation system

Start creating queries from new items (e.g. podcast title, author etc.) and ask LLM to rewrite as natural language query

## Unified Ranker for Search & Recsys @ Netflix

[Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn)](https://www.arxiv.org/abs/2408.10394)

Example of Stripe building a transformer based foundation model from sequence of transactions to identify fraud.

Problem: teams deal with complexity from bespoke models for search, similar item recs, pre-query recs
- High operational cost and missed transfer learning opportunities

Unified Contextual Ranker (UniCoRn) takes in a unified input schema and returns a prediction. Unified inputs:
- User ID
- item ID
- Search Query
- Task

Some clever tricks to reframe item to item recommendations as search, by using last item title as query.

Unified model used for search, pre-query filtering, video to video recs and more. Able to match or exceed previous task based models. Can iterate much faster with a unified model.

## Unified Embeddings @ Etsy

Problem: How to help users get better results with highly specific or broad queries, on ever-changing inventory.

- Query `mother's day gift` does not match product vocabulary
- Lexical retrieval does not account for user preferences

Solution: Unified embedding and retrieval model

- Two tower architecture for user and product side
- Add a quality vector on the product side (rating, freshness, conversion rate) concatenated to the product vector
- Add a constant vector on the user side just to make dimensions match





