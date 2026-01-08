# Miscellaneous Notes

A collection of miscellaneous, useful notes.


## f-strings

To surround a `text` with a symbol (say `=`) to a fixed length:

```python
>>> text = "Title"
>>> print(f"{text:=^20}")

=======Title========
```

## Vim

Command to interactively change each 'foo' to 'bar'. `:%s` triggers the substitute global command, followed by the search and replace phrases respectively. Finally `g` means replace all occurrences and `c` means with confirmation. Note that `:s` will only do the same for one line.

```bash
:%s/foo/bar/gc
```

## Find Files

To find files anywhere on the system with the filename `python` using bash, use:

```bash
find . -name python
```

We can add `*` before and/or after the filename to allow other characters before or after our keyword:

```bash
find . -name *python*
```

To search not just in the filename but also in the full path (e.g. we only want to search in `Desktop`), we can do:
```bash
find . -wholename "*Desktop*python*"
```

Note that if we want to locate executable binaries, another useful command is `whereis`:
```bash
whereis cat
---
cat: /usr/bin/cat /usr/share/man/man1/cat.1.gz
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

In the context of information retrieval, Trey Grainger in [AI-Powered Search](https://www.manning.com/books/ai-powered-search) suggests a relatedness measure to connect arbitrary entities together. Suppose we have a collection of jobs and each job is tagged with a set of skills. Suppose we wish to retrieve relevant skills to an arbitrary free text query $q$. 

The relatedness idea is to define a foreground of documents, e.g. based on a retrieval of documents using query $q$ which are related to the query, and to compare the attributes of the foreground against the background, i.e. all documents. 

Mathematically, we can think of the foreground documents as a sample, and the background documents as the population. The strength of the relationship between each skill $ t $ to the query $ q $ may then be defined as the z-statistic of the one-sample z-test of proportions of the occurrence of skill $ t $ in the foreground sample compared against the background population. A significantly greater occurrence in the sample compared to the population suggests a strong relationship between $ t $ and $ q $, and vice versa. Specifically:

$$
  z = \frac{ \hat{p} - p }{ \sqrt{ \frac{ p(1-p) }{n} } }
$$

Where:
- $ \hat{p} = \frac{ df(q, t) }{ df(q) } $ is the sample proportion.
- $ df(q, t) $ is the number of documents in the foreground corresponding to query $ q $ and contains skill $ t $.
- $ df(q) $ is the total number of documents in the foreground corresponding to query $ q $. It is also the number of samples $ n $.
- $ p = P(t) $ is the probability of skill t appearing across all documents.

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

## Memory Profiling

It is often useful to profile the memory usage of our script. In python, we can use `memory_profiler` to check the memory usage of our program line by line.

```python
from memory_profiler import profile
import sys


@profile(stream=sys.stdout)
def f():
    a = [1] * (10**6)
    b = [2] * (2 * 10**7)
    del b


if __name__ == "__main__":
    f()
```

This will print the following useful information to stdout. Note that even before we did anything, there is background memory usage of 17MiB.

```
Filename: memory_profiling.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     5     17.1 MiB     17.1 MiB           1   @profile(stream=sys.stdout)
     6                                         def f():
     7     24.5 MiB      7.5 MiB           1       a = [1] * (10**6)
     8    177.2 MiB    152.6 MiB           1       b = [2] * (2 * 10**7)
     9     24.8 MiB   -152.4 MiB           1       del b
```

We might also want to track memory usage of a function over time. We can use `memory_usage` instead for that.

```python
import time
from memory_profiler import memory_usage
def g(a, n: int = 100):
    time.sleep(1)
    b = [a] * n
    time.sleep(1)
    del b
    time.sleep(1)

if __name__ == "__main__":
    usage = memory_usage((g, (1,), {"n": int(1e7)}), interval=0.5)
    print(usage)
```

This will give us an array like so, showing the spike in memory in the middle of g.

```
[17.375, 17.5234375, 17.5234375, 19.34765625, 93.59765625, 93.59765625, 17.53125, 17.53125, 17.53125]
```

## CUDA

PyTorch may sometimes throw errors if the installed torch version does not match the installed CUDA version. To address this, we need to first check the CUDA version using the `nvcc` command:

```bash
/usr/local/cuda/bin/nvcc --version
---
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

Then install the correct version of torch:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## List Comprehension

When doing nested list comprehensions in python, the later loop becomes nested as the inner loop. So this:

```python
l = [i * j for i in range(4) for j in range(5)]
```

Is equivalent to:

```python
l = []
for i in range(4):
    for j in range(5):
        l.append(i*j)
```

This also explains why flattening a list of lists is of the following form:

```python
list_of_lists = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
flattened = [item for sublist in list_of_lists for item in sublist]
```

When we translate it into a nested for loop, we can see that the inner loop should indeed be placed at the end:

```python
flattened = []
for sublist in list_of_lists:
    for item in sublist:
        flattened.append(item)
```

## Configure VsCode Snippets

When typing stuff in VsCode, we sometimes write the same boilerplate and want to have a shortcut to type it for us. For example, when writing latex, we often have to create an `align` environment like so:
```bash
$$
\begin{align*}
    Equations go here..
\end{align*}
$$
```

VsCode has a snippets function to help us with that.
1. Press `Ctrl + Shift + P` and type `Configure User Snippets`
2. Create a local or global snippets file and name it (e.g. `charles.code-snippets`)
3. Add the snippet definition like the below

```bash
{
	"LaTeX Align Environment": {
	  "prefix": "align",
	  "body": [
	    "$$",
	    "\\begin{align*}",
	    "\t$1",
	    "\\end{align*}",
	    "$$$0"
	  ],
	  "description": "Inserts a LaTeX align* environment."
	}
}
```

Note that the prefix is the keyword to trigger the template. To use the template, we simply enter `insert` mode in vim as per normal (e.g. using `a` or `i` or `o`), then type `align`. At this point, we would see the text as per normal. We can then press `ctrl+space` to show the snippets auto-completion, if there are any. Finally, we just press `enter` to get the template generated. The cursor will now come to rest at position `$1` as indicated in the snippet.

## Github in Colab

We may want to run `.ipynb` notebooks within a github project in colab. There are two steps to this.

First, to get the `.ipynb` itself into colab, we can either use the UI to manually upload the notebook, or if we are loading from Github, we can simply start with the url to the notebook `https://github.com/charleslow/REPOSITORY/blob/master/NOTEBOOK.ipynb` and change `github` to `githubtocolab`. This will automatically open up the notebook in colab.

However, this only gives us the standalone notebook, and we will not have access to the project directory and packages. Hence, in the notebook, we need to:
- `!git clone <Insert REPOSITORY name>.git`
- `%cd folder name`

This will allow us to import packages and run the notebook as per normal.

## Quartz Cron

We often use quartz cron expressions to schedule jobs. A cron expression is simply 6 or 7 values in a space-separated string, such as:
```
"0 0 0 1 * ?"
```

Which means run on the:
- `0th second` of the
- `0th minute` of the
- `0th hour` of the
- `1st day the month` of
- `every month`
- and doesn't matter which `day of the week`

The specifics of each field are in the table below, from [the cron trigger tutorial](https://www.quartz-scheduler.org/documentation/quartz-2.3.0/tutorials/crontrigger.html):

| Field Name | Mandatory | Allowed Values | Allowed Special Characters |
| :--- | :--- | :--- | :--- |
| Seconds | YES | 0-59 | `, - * /` |
| Minutes | YES | 0-59 | `, - * /` |
| Hours | YES | 0-23 | `, - * /` |
| Day of month | YES | 1-31 | `, - * ? / L W` |
| Month | YES | 1-12 or JAN-DEC | `, - * /` |
| Day of week | YES | 1-7 or SUN-SAT | `, - * ? / L #` |
| Year | NO | empty, 1970-2099 | `, - * /` |

On the special characters:
- `*` is commonly used, meaning "select all the values"
- `?` is only used for Day of month or Day of week. Basically if we set one, we must set the other to `?` to avoid a conflict
- `-` is used to specify a range. E.g. `10-12` in the hour field means run on the `10th`, `11th` and `12th` hour
- `,` is used to specify a few values, e.g. `10,12` n the hour field means run on the `10th` and `12th` hour
- `/` is used to specify increments. e.g. `0/15` in the minutes field means run on minutes `0, 15, 30, 45`
- `#` is a special case for Day of the Week. Basically, `MON#1` means the 1st occurrence of a Monday of the given month

That's about it! Or we can just use an LLM to generate the expression we need.

## Wayland in Devcontainer

Sometimes we may encounter this error when trying to use vscode to setup a devcontainer:
```
Error response from daemon: invalid mount config for type "bind": bind source path does not exist: /run/user/1000/wayland-0
```

This happens because VS Code is trying to share the graphics connection with the container, which is not needed in most cases. Here is how to disable it:
- Open VS Code Settings `(Ctrl + ,)`
- Search for "Mount Wayland Socket"
- Uncheck the box for Dev > Containers: Mount Wayland Socket
- Rebuild your container (Ctrl + Shift + P -> "Rebuild Container").