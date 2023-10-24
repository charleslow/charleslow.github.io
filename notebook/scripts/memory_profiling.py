from memory_profiler import profile, memory_usage
import sys
import time


@profile(stream=sys.stdout)
def f():
    a = [1] * (10**6)
    b = [2] * (2 * 10**7)
    del b


def g(a, n: int = 100):
    time.sleep(1)
    b = [a] * n
    time.sleep(1)
    del b
    time.sleep(1)


if __name__ == "__main__":
    f()
    usage = memory_usage((g, (1,), {"n": int(1e7)}), interval=0.5)
    print(usage)
