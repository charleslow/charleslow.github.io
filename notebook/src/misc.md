# Miscellaneous Notes

A collection of miscellaneous, useful notes.

## Vim

Command to interactively change each 'foo' to 'bar'. `:%s` triggers the substitute global command, followed by the search and replace phrases respectively. Finally `g` means replace all occurrences and `c` means with confirmation. Note that `:s` will only do the same for one line.

```bash
:%s/foo/bar/gc
```

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

## Package Versioning

[link](https://python-poetry.org/docs/dependency-specification/)

[SemVer](https://semver.org/) is the versioning standard for Python packages. For a package version of e.g. `1.2.3`:
- The first number `1` is the major version. We update it when we make major API changes. A major of `0` indicates that the package is under initial development and anything may change.
- The second number `2` is the minor version. We update it when we add functionality in a backward compatible manner.
- The third number `3` is the patch version. We update it when we patch bugs in a backward compatible manner.

Poetry is a useful tool to manage package dependencies in a Python library. In the `pyproject.toml` file, we specify package dependencies in the following manner:

```
[tool.poetry.dependencies]
python = ">=3.8,<3.11"
requests = "^2.26.0"
pytz = "~2022.1"
```

The `caret` requirement (e.g. `requests = "^2.26.0"`) means that SemVer compatible changes are allowed, i.e. an update is allowed if the version number does not modify the left-most non-zero digit. E.g.:
- `^1.2.3` means that `1.3.0` is allowed but not `2.0.0`
- `^0.2.4` means that `0.2.5` is allowed but not `0.3.0`

The `tilde` requirement is somewhat stricter. It specifies a minimal version but allows room to upgrade depending on how many digits are supplied. If `major.minor.patch` or `major.minor` is supplied, only patch-level changes are allowed. If `major` is supplied, minor and patch level changes are allowed.
- `~1.2.3` means that `1.2.7` is allowed but not `1.3.0`
- `~1.2` means that `1.2.7` is allowed but not `1.3.0`
- `~1` means that `1.3.0` is allowed but not `2.0.0`

The poetry versioning specification depends on developer updating the library versions in a disciplined way. It allows us to provide some flexibility in package versioning while avoiding updating a dependent package to a version that breaks our code. Having more flexibility in specifying package dependencies reduces the risk of dependencies version conflicting.

## Parallelism in Python

References: [realpython](https://realpython.com/python-gil/), [superfastpython](https://superfastpython.com/numpy-vs-gil/)

The Python Global Interpreter Lock (GIL) is a lock that allows only one thread to run at a time. The reason for this design feature in Python is because the Python interpreter is not thread-safe, meaning that if the GIL did not exist, the Python interpreter could introduce race conditions and other fatal errors due to multiple threads modifying memory at the same time. The GIL also allowed other non-thread-safe C programs to be easily integrated into the Python ecosystem, which contributed to Python's success as a language.

The GIL also improves performance for single-threaded programs, as it only requires a single lock to be managed (not sure how this works). There have been attempts to remove the GIL from Python but none have been successful because they would degrade single-threaded performance.

For many C extensions (e.g. numpy), multi-threading is still possible because these extensions are allowed to manually release the GIL (as long as they restore things back to normal when the functions return). This allows us to still use multi-threading for CPU-intensive functions with the GIL around. Similarly for Rust, we can release the GIL to achieve [parallelism](https://pyo3.rs/v0.9.2/parallelism).

Alternatively, we can use `multiprocessing` to create multiple processes (instead of threads). Each process contains its own Python interpreter (and GIL) and hence can run truly in parallel. The downside is that the overhead of creating and managing processes is much more than that for threads, meaning that the benefits of multiprocessing are much dampened compared to multi-threading.

## LightGBM Memory

I often run into memory issues running LightGBM. So here are some experiments to measure memory usage and understand how hyperparameters can affect memory usage.

The function of interest is the `fit` method for the learn to rank task.

```python
import lightGBM as lgb
def f():
    model = lgb.LGBMRanker(**params, objective="lambdarank")
    model.fit(
        X=data,
        y=y,
        group=groups,
    )
```

The memory usage is measured using the `memory_profiler` module, which checks the memory usage at .1 second intervals. The maximum is then taken to represent the maximum memory usage of the fit function. We also take note of the size of the data itself (using `data.nbytes`) and subtract that away to get closer to the LightGBM memory usage.

```python
from memory_profiler import memory_usage
def run(params):
    mem_usage = memory_usage(f)
    return max(mem_usage) / 1000 # GB
```

We set the default parameters as follows and generate the data this way. For the experiments below, the default parameters are used unless specified otherwise.

```python
DEFAULT_PARAMS = {
    "N": 200000, # number of instances
    "M": 500, # feature dimension
    "n_estimators": 100,
    "num_leaves": 100,
    "histogram_pool_size": -1,
}
data = np.random.randn(DEFAULT_PARAMS["N"], DEFAULT_PARAMS["M"])
groups = [20] * int(N / 20) # assume each session has 20 rows
y = np.random.randint(2, size=N) # randomly choose 0 or 1
```

Large `num_leaves` can get very memory intensive. We should not need too many leaves, so generally using `num_leaves <= 100` and increasing the number of estimators seems sensible to me.

- num_leaves: `10`, Maximum memory usage: 2.28 GB - 0.80 GB = `1.48 GB`
- num_leaves: `100`, Maximum memory usage: 2.52 GB - 0.80 GB = `1.72 GB`
- num_leaves: `1000`, Maximum memory usage: 4.04 GB - 0.80 GB = `3.24 GB`

Increasing `n_estimators` doesn't seem to raise memory much, but increases run time because each tree is fitted sequentially on the residual errors, so it cannot be parallelized.

- n_estimators: `10`, Maximum memory usage: 2.28 GB - 0.80 GB = `1.48 GB`
- n_estimators: `100`, Maximum memory usage: 2.53 GB - 0.80 GB = `1.73 GB`
- n_estimators: `1000`, Maximum memory usage: 2.69 GB - 0.80 GB = `1.89 GB`

Increasing `N` increases memory sublinearly. It seems that the data size itself will be more of a problem than the increase in LightGBM memory usage as `N` increases. For extremely large `N`, we can also set the `subsample` parameter to use only a fraction of the training instances for each step (i.e. stochastic rather than full gradient descent). By default `subsample=1.0`.

- N: `1,000`, Maximum memory usage: 0.38 GB - 0.00 GB = `0.38 GB`
- N: `10,000`, Maximum memory usage: 0.45 GB - 0.04 GB = `0.41 GB`
- N: `100,000`, Maximum memory usage: 1.46 GB - 0.40 GB = `1.06 GB`
- N: `1,000,000`, Maximum memory usage: 6.12 GB - 4.00 GB = `2.12 GB`
- N: `2,000,000`, Maximum memory usage: 10.48 GB - 8.00 GB = `2.48 GB`

In contrast to `N`, memory usage is quite sensitive to `M`, seems to increase linearly when `M` gets large. `M=10,000` blows up my memory. I suppose this could be mitigated by setting `colsample_bytree` or `colsample_bynode` to sample a smaller subset.
- M: `100`, Maximum memory usage: 2.08 GB - 0.16 GB = `1.92 GB`
- M: `1000`, Maximum memory usage: 4.92 GB - 1.60 GB = `3.32 GB`
- M: `2000`, Maximum memory usage: 9.69 GB - 3.20 GB = `6.49 GB`
- M: `3000`, Maximum memory usage: 14.35 GB - 4.80 GB = `9.55 GB`

To deal with the high memory usage of large `M`, we can set `colsample_bytree` which samples a subset of columns before training each tree. This will help to mitigate the memory usage. For this experiment, we set `M=2000` to simulate data with high number of dimensions.

- `colsample_bytree`: 0.1, Maximum memory usage: 8.60 GB - 3.20 GB = `5.40 GB`
- `colsample_bytree`: 0.2, Maximum memory usage: 9.58 GB - 3.20 GB = `6.38 GB`
- `colsample_bytree`: 0.4, Maximum memory usage: 10.06 GB - 3.20 GB = `6.86 GB`
- `colsample_bytree`: 0.6, Maximum memory usage: 10.07 GB - 3.20 GB = `6.87 GB`
- `colsample_bytree`: 0.8, Maximum memory usage: 10.46 GB - 3.20 GB = `7.26 GB`

In contrast, setting `colsample_bynode` does not help memory usage at all. Not too sure why, but I suppose since multiple nodes for the same tree can be split at the same time, the full feature set still has to be kept in memory.
- `colsample_bynode`: 0.1, Maximum memory usage: 10.49 GB - 3.20 GB = `7.29 GB`
- `colsample_bynode`: 0.2, Maximum memory usage: 10.49 GB - 3.20 GB = `7.29 GB`
- `colsample_bynode`: 0.4, Maximum memory usage: 10.49 GB - 3.20 GB = `7.29 GB`
- `colsample_bynode`: 0.6, Maximum memory usage: 10.49 GB - 3.20 GB = `7.29 GB`
- `colsample_bynode`: 0.8, Maximum memory usage: 10.48 GB - 3.20 GB = `7.28 GB`