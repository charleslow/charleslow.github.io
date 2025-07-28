# MinGPT

This is a walkthrough of KArpathy's MinGPT implementation

## `bpe.py`

This file contains code to implement a Byte Pair Encoding encoder. It does not contain code for training, just loads openai's GPT-2 bpe encoding for inference.

Most text is represented in UTF-8 encoding, which is just a sequence of bytes (values `0 to 255`). For example, `0xf0` corresponds to `33` in decimal which corresponds to the character `!`. This means that all text can be treated as a sequence of byte values.

As a fallback, we first need token representations for individual byte values (in case we encounter out of vocab tokens with unknown byte sequences). 

First, two files are downloaded in `get_encoder`, from `https://openaipublic.blob.core.windows.net/gpt-2/models/124M/`:
- `encoder.json` is a dict of len `50257` mapping from a token to its index. This represents the entirety of the vocabulary
    - The first `256` tokens represents the `256` byte values. Each token is some arbitrarily chosen character (just need to make sure it is printable)
    - These tokens are a *fallback* to ensure that we can encode any text sequence. For example, if we encounter a new emoji with unknown byte sequence, at the very least we can encode each byte separately.
    - The next `50k` tokens map from a byte sequence of length `2` and above to an index, these are the BPE mined sequences of merged bytes
    - The last token is `<|endoftext|>` which is a special token
- `vocab.bpe` is a `\n` separated list of byte sequences that should be merged (`50k` of them)
    - In contrast to the above, these sequences are not merged yet (e.g. a line is `R ocket`)
    - We store these as a `list[tuple]` in `bpe_merges`

These two data are passed into the `Encoder` main class.
