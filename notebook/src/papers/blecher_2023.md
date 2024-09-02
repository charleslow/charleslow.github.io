# Blecher 2023 - Nougat

[Blecher 2023 - Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/pdf/2308.13418).

Nougat is an end-to-end system that converts a scientific PDF into a sequence of tokens in markdown format in an auto-regressive way.

Prior methods for Visual Document Understanding (VDU) usually rely on an external OCR service to generate intermediate outputs. In contrast, this method is end-to-end, and the text is generated directly from image embeddings in a decoder manner. Thus, the model is very simple and most of the work in this paper is in data preparation.

## Model

1. Encoder. The encoder gets a variable size document image $x \in \R^{3 \times H_0 \times W_0}$ and applies crop / resizing to generate a fixed rectangle of size $(H, W)$. Smaller images are white padded. The fixed size image can then be passed into a Swin Transformer to output a sequence of embedded patches $z \in \R^{d \times N}$ where $d$ is the latent dimension and $n$ is the number of patches.


##
