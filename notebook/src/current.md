# Current Focus

## 2025 Goals

- GNN-based approach for search and recommendation
    - Use past sequential history of items as user representation
    - Transformer-based reranking
    - Transformer-based user encoding for ANN retrieval

- Multi-task, multi-purpose embeddings
    - For retrieval and reranking
    - Across various services (jobs, courses, skills)

## Research

- Optimizing LLM explanations based on implicit feedback
    - How to optimize an LLM to provide better recommendation explanations by fine-tuning on implicit feedback?

- Replacing BM25
    - How to design a search system that matches BM25 performance at cold start and gradually improves with more data, without dropping below BM25 performance?

- Precise Retrieval
    - The common two tower approach to embedding retrieval leaves much to be desired
        - There is no natural score threshold at which items are deemed irrelevant. Traditionally, classifiers have a `0.5` score cut-off.
        - Embedding retrieval tends to retrieve unrelated items. This is a well documented problem. For example, `Nike shoes` retrieves `Adidas Shoes`. 