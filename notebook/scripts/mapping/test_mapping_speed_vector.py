"""
A continuation from test_mapping_speed.py
But instead of encoding a list of strings, we have a np.array of strings with n_rows and n_cols.

Conclusion: np.vectorize is faster.

>>> main(num_rows=5000, num_cols=25, num_vocab=200, num_trials=100)
Method1: 41.303ms
Method6: 24.230ms

>>> main(num_rows=500, num_cols=1, num_vocab=200, num_trials=100)
Method1: 0.177ms
Method6: 0.095ms
"""
import timeit
import random
import numpy as np
from copy import deepcopy
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
    return {v: float(i) for i, v in enumerate(sorted(vocab))}


def make_string(vocab: list[str]):
    return random.choice(vocab)


class Method1:
    """Simple List comprehension"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def __call__(self, S: np.ndarray):
        new_S = []
        for j in range(S.shape[1]):
            new_S.append([self.vocab.__getitem__(s) for s in S[:, j]])
        return np.array(new_S).T.astype(np.float32)


class Method6:
    """Use np.vectorize"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.vectorizer = np.vectorize(self.vocab.__getitem__)

    def __call__(self, S: np.ndarray):
        return self.vectorizer(S).astype(np.float32)


def main(
    num_rows: int = 500, num_cols: int = 15, num_vocab: int = 30, num_trials: int = 100
):
    vocab = make_vocab(num_vocab)
    vocab_keys = list(vocab)
    S = [[make_string(vocab_keys) for _ in range(num_rows)] for _ in range(num_cols)]
    S = np.array(S).T
    results = []
    for Method in [Method1, Method6]:
        # Get result
        method = Method(vocab)
        res = method(S)
        results.append(res)

        # Do benchmark
        timing = (
            timeit.timeit(
                lambda: method(S),
                number=num_trials,
            )
            / num_trials
        )
        print(f"{type(method).__name__}: {timing*1000:.3f}ms")

    for i in range(1, len(results)):
        assert (results[i - 1] == results[i]).all()


main(num_rows=500, num_cols=1, num_vocab=200, num_trials=100)
