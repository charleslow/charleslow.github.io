import lightgbm as lgb
import numpy as np
from memory_profiler import memory_usage
import matplotlib.pyplot as plt


def make_data(n: int, m: int, num_sessions: int):
    assert n % num_sessions == 0, "n should divide num_sessions"
    X = np.random.randn(n, m)
    groups = [num_sessions] * int(n / num_sessions)
    y = np.random.randint(2, size=n)  # randomly choose 0 or 1
    return X, groups, y


def f(X, groups, y):
    model = lgb.LGBMRanker(objective="lambdarank")
    model.fit(
        X=X,
        y=y,
        group=groups,
    )


def plot(mem_usage, title: str):
    plt.plot(mem_usage)
    plt.xlabel("Iteration in .1 second intervals")
    plt.ylabel("Memory usage (MB)")
    plt.title(title)
    plt.savefig("lgb_memory.png")
    plt.close()


if __name__ == "__main__":
    X, groups, y = make_data(1000_000, 500, 50)
    mem_usage = memory_usage((f, (X, groups, y)), interval=0.1)
    plot(mem_usage, "LightGBM memory usage")
