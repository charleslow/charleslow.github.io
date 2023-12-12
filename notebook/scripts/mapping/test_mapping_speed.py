"""
We wish to investigate the fastest way to encode a list of strings into a list of ints.
This is for the purpose of converting categorical features into an integer mapping that lightgbm can consume.

Conclusion: Simple list comprehension is still the fastest.

>>> main(500, 50)
Method1: 0.003486s
Method2: 0.004537s
Method3: 0.109468s
Method4: 0.076830s
Method5: 0.029698s
Method6: 0.006123s
"""
import timeit
import random
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pdb import set_trace

LETTERS = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
]


def make_vocab(num_vocab: int):
    vocab = [
        "".join([random.choice(LETTERS) for _ in range(random.randint(1, 10))])
        for _ in range(num_vocab)
    ]
    return {v: i for i, v in enumerate(sorted(vocab))}


def make_string(vocab: list[str]):
    return random.choice(vocab)


class Method1:
    """Simple List comprehension"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def __call__(self, strings: list[str]):
        return [self.vocab.get(s, None) for s in strings]


class Method2:
    """List with map"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def __call__(self, strings: list[str]):
        return list(map(lambda s: self.vocab.get(s, None), strings))


class Method3:
    """tokenizers encode_batch"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def __call__(self, strings: list[str]):
        return [encoding.ids[0] for encoding in self.tokenizer.encode_batch(strings)]


class Method4:
    """tokenizers encode with list comp"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def __call__(self, strings: list[str]):
        return [self.tokenizer.encode(s).ids[0] for s in strings]


class Method5:
    """tokenizers encode with string joining"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def __call__(self, strings: list[str]):
        return self.tokenizer.encode(" ".join(strings)).ids


class Method6:
    """Use np.vectorize"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.vectorizer = np.vectorize(self.vocab.__getitem__)

    def __call__(self, strings: list[str]):
        return self.vectorizer(strings).tolist()


def main(n: int = 500, num_vocab: int = 30):
    vocab = make_vocab(num_vocab)
    vocab_keys = list(vocab)
    strings = [make_string(vocab_keys) for _ in range(n)]
    results = []
    for Method in [Method1, Method2, Method3, Method4, Method5, Method6]:
        method = Method(vocab)
        res = method(strings)
        results.append(res)
        timing = timeit.timeit(lambda: method(strings), number=100)
        print(f"{type(method).__name__}: {timing:.6f}s")

    for i in range(1, len(results)):
        assert results[i - 1] == results[i]


main(500, 50)
