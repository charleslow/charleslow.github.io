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

## asyncio

[reference](https://realpython.com/async-io-python/)

asyncio is a single-threaded framework that does not use multi-threading or multi-processing to speed up tasks. Instead, a coordinator (or event loop) passes control from a time-consuming blocking function (e.g. time.sleep or an I/O operation) to other functions to run. This passing of control occurs with the `await` keyword. When the blocking function is completed, it notifies the coordinator and control returns to where the blocking function left off.

`asyncio` does not speed up CPU-bound tasks, due to its single-threaded design. It only works when the function being awaited is an I/O operation that is supported by asyncio. This includes [stuff](https://realpython.com/async-io-python/#libraries-that-work-with-asyncawait) like:
* HTTP (supported through `aiohttp`)
* DB calls (e.g. `aioredis`)

`asyncio` is preferred for such tasks (e.g. making multiple I/O calls and doing something when they all return) over multi-processing because of its single-thread design, making it easier to debug.


## relatedness

In the context of information retrieval, Trey Grainger in [AI-Powered Search](https://www.manning.com/books/ai-powered-search) suggests a relatedness measure to connect arbitrary entities together. Suppose we have a collection of jobs and each job is tagged with a set of skills. Suppose we wish to retrieve relevant skills to an arbitrary free text query \\( q \\). 

The relatedness idea is to define a foreground of documents, e.g. based on a retrieval of documents using query \\( q \\) which are related to the query, and to compare the attributes of the foreground against the background, i.e. all documents. 

Mathematically, we can think of the foreground documents as a sample, and the background documents as the population. The strength of the relationship between each skill \\( t \\) to the query \\( q \\) may then be defined as the z-statistic of the one-sample z-test of proportions of the occurrence of skill \\( t \\) in the foreground sample compared against the background population. A significantly greater occurrence in the sample compared to the population suggests a strong relationship between \\( t \\) and \\( q \\), and vice versa. Specifically:

\\[
  z = \frac{ \hat{p} - p }{ \sqrt{ \frac{ p(1-p) }{n} } }
\\]

Where:
- \\( \hat{p} = \frac{ df(q, t) }{ df(q) } \\) is the sample proportion.
- \\( df(q, t) \\) is the number of documents in the foreground corresponding to query \\( q \\) and contains skill \\( t \\).
- \\( df(q) \\) is the total number of documents in the foreground corresponding to query \\( q \\). It is also the number of samples \\( n \\).
- \\( p = P(t) \\) is the probability of skill t appearing across all documents.

By performing a retrieval and ranking skills based on the z-statistic, we can compute a relationship between any arbitrary query and attribute of the documents (on the fly, if necessary). This functionality is implemented in [solr](https://solr.apache.org/guide/8_5/json-facet-api.html#relatedness-and-semantic-knowledge-graphs).