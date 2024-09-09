# Packages

## KeyBERT, KeyLLM

References:
- [KeyBERT Article](https://www.maartengrootendorst.com/blog/keybert/)
- [KeyLLM Article](https://www.maartengrootendorst.com/blog/keyllm/)

<<KeyBERT>> and <<KeyLLM>> are packages to perform unsupervised keyword extraction from text. KeyBERT relies on BERT-based models, and the main idea is to extract n-gram phrases which have high semantic similarity to the overall document embedding. Some additional features are:
- Allow user to specify phrase length for extraction
- Add diversification via MMR to get diverse phrases

KeyLLM taps on LLMs to enhance the keyword extraction. Basically, it creates a prompt to ask an LLM to extract keywords from a document. It integrates with KeyBERT such that we can use KeyBERT to cluster documents, and only run KeyLLM on one document per cluster to save costs. It can also use KeyBERT to suggest candidates and use the LLM to verify.