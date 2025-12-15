# Current Focus

## 2025 Goals

Incorporate semantic IDs into a generalizable search and recommender system:
- Can function well with just a system prompt (guiding the overall goal of the system) and an item catalogue
- Handles a changing catalogue gracefully
- Good search and recommendation latency
- Cheap (runs on a T4 or equivalent)

## Research Questions

- Optimizing LLM explanations based on implicit feedback
    - How to optimize an LLM to provide better recommendation explanations by fine-tuning on implicit feedback?

- Replacing BM25
    - How to design a search system that matches BM25 performance at cold start and gradually improves with more data, without dropping below BM25 performance?

- Precise Retrieval
    - The common two tower approach to embedding retrieval leaves much to be desired
        - There is no natural score threshold at which items are deemed irrelevant. Traditionally, classifiers have a `0.5` score cut-off.
        - Embedding retrieval tends to retrieve unrelated items. This is a well documented problem. For example, `Nike shoes` retrieves `Adidas Shoes`. 
        - Can we have embedding models that approximate AND / OR conditions that more naturally fit into the retrieval paradigm?

- Efficient learning of semantic IDs
    - LLMs can learn Semantic IDs as part of their language and recommend and reason about items once they learn the "language" using Supervised Fine Tuning
    - But the danger is catastrophic forgetting and losing capabilities as they get fine tuned in this way
    - Is there a more "natural" way for LLMs to learn such semantic IDs?
    - How do we handle a changing catalogue in real-time gracefully?