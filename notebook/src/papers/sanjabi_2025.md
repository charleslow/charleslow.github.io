# Sanjabi 2025 - 360Brew

[360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://arxiv.org/pdf/2501.16450)

360Brew is a 150B foundational LLM model used to centralize many LinkedIn models into one model.

For V1.0, 360Brew focused on replacing ranking tasks since they are less bounded by computational constraints in practice compared to retrieval. 

## Current Paradigm

The current paradigm relies on bloom embedding of ID based features for both members and items. These large embedding tables are supplemented with other engineered features such as item attributes. Some challenges with the current paradigm are:
- Cold start: ID based features cannot handle cold start, and thus a lot of work is done to learn good content-based representations
- Feature interactions: Previously, 
