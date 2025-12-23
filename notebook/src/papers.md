Summaries of individual papers that I have taken a deeper look into.

# Quick Reads

Here I put down quick summaries of papers I read quickly. May possible revisit these with its own page if a deep dive is warranted.

## [Assran 2023 - Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)

This is the I-JEPA paper. This paper aims to perform JEPA-style self-supervised learning for vision transformers without using image augmentations (like rotation, shear etc.). The main idea is to take a random image patch as <<context>> and aim to predict multiple other image patches as <<target>>. Importantly:
- The loss is to minimize l2-distance between the $\text{Predictor}(\text{Context Encoding})$ and $\text{Target Encoding}$ in latent space
- To avoid representation collapse in latent space, the target encoder is updated as a trailing exponential moving average of the Context Encoder
- This approach is simple and scalable and matches performance of SOTA methods using image augmentations

