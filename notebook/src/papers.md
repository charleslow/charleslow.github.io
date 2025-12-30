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