# Information Theory Roadmap

A roadmap for learning information theory and how it applies to model compression and hallucinations.

## Week 1: Foundations of Information Theory

- Mackay Chapters 1 (Intro), 2 (Probability / Entropy), 4 (Source coding theorem)
- Check that minimizing cross entropy loss is identical to minimizing KL divergence

## Week 2: Practical Compression with Arithmetic Coding

- Mackay Chapter 6 (stream codes)
- Implement a basic arithmetic coder to compress a string given a fixed probability table

## Week 3: Bayesian Model Comparison

- Mackay Chapter 28 (model comparison)

## Week 4: Concentration Inequalities

- High dimensional probability (Vershynin) Chapters 2-3.

## Week 5: Universal Coding and MDL

- Cover and Thomas Chapter 13 (Universal source coding)
- Grunwald Chapters 1-3 (introduction, Kolmogorov complexity, universal coding)

## Week 6: Channel Coding and Mutual Information

- Mackay Chapter 9 (channel capacity)
- Cover and Thomas Chapter 2 (entropy, mutual information properties)

## Week 7: Compression and LLM Hallucinations

- Read paper [Predictable Compression Failures: Why Language Models Actually Hallucinate](https://arxiv.org/abs/2509.11208)

## Week 8: Rate Distortion and Information Bottleneck

- Cover and Thomas Chapter 10 (rate distortion)
- Tishby 1999 - The information bottleneck method
- Alemi 2017 - Deep variational information bottleneck

## Week 9: Neural Compression (Optional)

- Han 2016 - Deep Compression
- Practical LLM quantization (GPTQ, AWQ)

## References

1. **Book:** *Information Theory, Inference, and Learning Algorithms* - David MacKay.
2. **Book:** *Elements of Information Theory* - Cover & Thomas
3. **Monograph:** *High-Dimensional Probability* - Vershynin
4. **Paper:** *Predictable Compression Failures* (arXiv:2509.11208).
5. **Book:** *The minimum description length principle* - Grunwald.
