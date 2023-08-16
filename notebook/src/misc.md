# Miscellaneous Notes

A collection of miscellaneous, useful notes.

## Numpy Indexing

Suppose we have a 2D array `X` and would like to take a slice of certain rows and columns. We might try, for e.g., to take the first two rows of X and the 3rd/4th column of X, i.e. we expect to get a 2 by 2 matrix.

```python
import numpy as np
X = np.random.randn(5, 5)
X[[0, 1], [2, 3]]
```

Unfortunately, this will return an array of items `array(X[0, 2], X[1, 3])`, which is not what we want. Instead, a slightly inefficient but clear way is to slice each axis separately. It is not optimal because the first slice creates a temporary array view before the second slice is applied.

```python
X[[0, 1], :][:, [2, 3]]
```

Finally, the recommended way seems to be to use `np.ix_`. Doing `np.ix_([0, 1], [2, 3])` creates a tuple of two elements.

```python
idx = np.ix_([0, 1], [2, 3])
idx
>> (array([[0],
           [1]]),
>>  array([[2, 3]]))
```

Indexing the original array with this output `X[idx]` will then give us what we want.