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