import lightgbm as lgb
from flaml import AutoML, tune
import numpy as np
from memory_profiler import memory_usage
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def make_data(n: int, m: int, num_sessions: int, dtype: str):
    assert n % num_sessions == 0, "n should divide num_sessions"
    if dtype == "boolean":
        X = np.random.randint(2, size=(n, m))
    elif dtype == "category":
        X = np.random.randint(50, size=(n, m))
    else:
        X = np.random.randn(n, m)

    cols = [f"c{i}" for i in range(m)]
    groups = [num_sessions] * int(n / num_sessions)
    y = np.random.randint(2, size=n)  # randomly choose 0 or 1
    return pd.DataFrame(X, columns=cols).astype({col: dtype for col in cols}), groups, y


def f(X, groups, y, method="lgbm"):
    if method == "lgbm":
        model = lgb.LGBMRanker(objective="lambdarank")
        model.fit(
            X=X,
            y=y,
            group=groups,
        )
    else:
        model = AutoML(estimator_list=["lgbm"])
        model.fit(
            X_train=X,
            y_train=y,
            groups=groups,
            task="rank",
            metric="ndcg",
            time_budget=10,
            # Settings to reduce OOM error
            # https://microsoft.github.io/FLAML/docs/FAQ/#how-to-resolve-out-of-memory-error-in-automlfit
            model_history=False,
            skip_transform=False,
            free_mem_ratio=0.1,
        )


def plot(mem_usage, title: str):
    plt.plot(mem_usage)
    plt.xlabel("Iteration in .1 second intervals")
    plt.ylabel("Memory usage (MB)")
    plt.title(title)
    plt.savefig("lgb_memory.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("lightgbm_memory")
    parser.add_argument("n", help="Number of data points.", type=int)
    parser.add_argument("m", help="Number of features.", type=int)
    parser.add_argument("n_sessions", help="Number of data in one session.", type=int)
    parser.add_argument("dtype", help="Type of data, e.g. category or Int16.", type=str)
    parser.add_argument("method", help="Either lgbm or automl", type=str)
    args = parser.parse_args()

    data_tuple = make_data(
        n=args.n, m=args.m, num_sessions=args.n_sessions, dtype=args.dtype
    )
    mem_usage = memory_usage((f, data_tuple + (args.method,)), interval=0.1)
    plot(mem_usage, "LightGBM memory usage")
