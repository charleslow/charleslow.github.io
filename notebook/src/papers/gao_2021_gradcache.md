# Gao 2021 - GradCache

[Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983)

This paper demonstrates a method to perform contrastive learning for two-tower model training with an arbitrarily large batch size at a constant memory cost, at the expense of slightly longer compute time.

## Background

The typical contrastive learning setup for learning a two tower retrieval model is explained by [Karpukhin 2020 - Dense Passage Retrieval](https://arxiv.org/abs/2004.04906), in which we have batches of related `(anchor, positive)` passages. Contrastive learning sets out to maximize the similarity between the anchor and positive passage, and minimize the similarity between the anchor and all other passages in the mini batch. It has been consistently shown that using larger batch sizes is critical to the performance of this training method, as the number of negatives increases, providing more information to the contrastive learning process.

However, using large batch sizes is impractical for most researchers, since the memory cost scales linearly with the batch size. The DPR paper used 8x V100 GPUs to process a batch size of 128, which is not attainable for most outfits. Thus the method in this paper is of great practical significance.

## Setup

We start with two classes or sets of data, $\mathcal{S}$ and $\mathcal{T}$. Typically $\mathcal{S}$ may represent a set of string queries and $\mathcal{T}$ may represent a set of document texts. We want to learn encoders $f$ and $g$ such that, given $s \in \mathcal{S}$ and $t \in \mathcal{T}$, the encoded representations $f(s)$ and $g(t)$ are close if related and far apart if not related. 

Typically, we set up a contrastive loss as follows. Sample a mini batch of anchors $S \subset \mathcal{S}$ and corresponding targets $T \subset \mathcal{T}$, where each element $s_i \in S$ has a corresponding target $t_{r_i} \in T$. The rest of the random samples in $T$ will be used as in-batch negatives.

We have an InfoNCE loss as follows:
$$
    \L = -\frac{1}{|S|} \sum_{s_i \in S} \log \frac{
        \exp(f(s_i))^T g(t_{r_i}) / \tau
    }{
        \sum_{t_j \in T} \exp(f(s_i))^T g(t_j) / \tau
    }
$$

Let us also denoted the parameters of $f$ as $\Theta$ and parameters of $g$ as $\Lambda$.

## Analysis

Now we show how the gradient computation and therefore training can be broken down to mitigate the memory bottleneck. Importantly, note that the main bottleneck in such contrastive training is that increasing the batch size scales linearly with the maximum memory requirement of the forward pass of the large BERT model. This is because we encode all texts in the mini-batch simultaneously and run backpropagation. Hence, we want a method that allows us to batch the forward pass within a mini-batch into mini-mini-batches (lets call it a tiny batch) while still allowing us to get the correct backpropagation gradients.

Applying the multivariate chain rule to the loss above, we have that:
$$
\begin{align*}
    \frac{\partial \L}{\partial \Theta} &=
        \sum_{s_i \in S} 
            \frac{\partial \L}{\partial f(s_i)}
            \frac{\partial f(s_i)}{\partial \Theta}\\
    \frac{\partial \L}{\partial \Lambda} &=
        \sum_{t_j \in T} 
            \frac{\partial \L}{\partial g(t_j)}
            \frac{\partial g(t_j)}{\partial \Lambda}\\
\end{align*}
$$

From these simple statements, the paper makes two important observations:
1. The partial derivative $\frac{\partial f(s_i)}{\partial \Theta}$ only depends on $\Theta$ and $s_i$. It does not depend on any other anchor or passage. Thus, if we have access to the numerical value of $\frac{\partial \L}{\partial f(s_i)}$, we can run backpropagation for $\frac{\partial \L}{\partial \Theta}$ independently from all other samples in an arbitrarily small batch.
2.  The partial derivative $\frac{\partial \L}{\partial f(s_i)}$ requires only the numerical values of the encoded representations $f(s_i)$ for all $s_i \in S$ and $g(t_j)$ for all $t_j \in T$. To compute these values, we don't actually need the computation graph states of the encoder $f$, we just need the numerical values of all the embeddings.

Note that we can 

The above statements are focused on $f$, $s_i$ and $\Theta$, but similar statements hold for $g$, $t_j$ and $\Lambda$. The first statement above allows us to <<run the expensive gradient updates on a small batch of anchors or passages at a time>>, which avoids the memory bottleneck of running gradient updates on a large batch size for the large encoders. We can do this so long as we have access to the partial derivatives $\frac{\partial \L}{\partial f(s_i)}$. The second statement shows us that computing these partial derivatives is not difficult, because <<we just need the encoded representations of each anchor and passage>>. Hence we can batch encode all the anchors and passages in the mini-batch (without gradients), and then use these values to compute this derivative.

## Method

The above explanation directly informs the algorithm called <<GradCache>>. It works as follows. First, split the large batch into tiny batches which can fit into memory, denoted as $S = \{ \hat{S_1}, \hat{S_2}, ... \}$ and $T = \{ \hat{T_1}, \hat{T_2} \}$.

<<Step 1: Graph-less Forward>>. We run a no-gradient forward pass of each encoder to obtain $f(s_i), g(t_j)$ for all $s_i \in S, t_j \in T$. We store all the encoded representations in memory.

<<Step 2: Gradient Caching>>. Using the pre-computed representations in step 1, we run a forward pass to obtain the loss $\L$. We then allow the autograd library to run a backward pass to get gradients $\frac{\partial L}{\partial f(s_i)}, \frac{\partial L}{\partial g(t_j)}$ for each representation $f(s_i)$ or $g(t_j)$. Note that we are not involving the encoders in this step at all, so memory costs are minimal (just some dot products).
- Denote $u_i = \frac{\partial \L}{\partial f(s_i)}$
- Denote $v_i = \frac{\partial \L}{\partial g(t_i)}$
- Store the representation gradient cache $\{ u_i, u_2, ..., v_1, v_2, ... \}$

<<Step 3: Tiny Batch Gradient Accumulation>>. Recall earlier we said that so long as we have access to the partial derivatives from the loss to the embeddings, we can compute gradients for each $s_i$ or $t_j$ in arbitrarily tiny batches. This is the step where we do so. Specifically, we perform <<gradient accumulation>> one tiny batch at a time.

For the parameters $\Theta$ of encoder $f$:
$$
\begin{align*}
    \frac{\partial \L}{\partial \Theta} 
        &= \sum_{\hat{S_j} \in S} \sum_{s_i \in \hat{S_j}}
            \frac{\partial \L}{\partial f(s_i)} \frac{\partial f(s_i)}{\partial \Theta}\\
        &= \sum_{\hat{S_j} \in S} \sum_{s_i \in \hat{S_j}}
            u_i \frac{\partial f(s_i)}{\partial \Theta}
\end{align*}
$$

The $u_i$s are simply looked up from the gradient cache. We perform encoder forward on a tiny batch at each time, multiply with $u_i$ and accumulate the gradients. Thus the memory requirement is limited to the encoder forward on a tiny batch, which can be arbitrarily small. Note that the final gradient computed and applied will be *equivalent to* the original gradients had we directly computed the loss for the large batch. We can see this from the double summation in the equation above, which simply equates to summing over all $s_i \in S$.

Finally, we perform a similar gradient accumulation for the parameters $\Lambda$ of encoder $g$. Once all the sub-gradients are accumulated, the optimizer step is taken to update model parameters as though we had processed the full batch in a single forward backward pass.

## Results

The results are based on replicating the Dense Passage Retrieval paper results using a smaller GPU. Note that DPR was trained on 8 GPUs and batch size of 128.
- DPR had a top-20 hit rate of `78.4`
- Using a batch size of 8, which was the largest batch size that fits in memory on an RTX 2080ti, the top-20 hit rate was `77.2`
- Using gradcache to simulate the batch size of 128, the top-20 hit rate was `79.3`
- Using gradcache to simulate a larger batch size of 512, the top-20 hit rate was `79.9`

These results demonstrate the importance of a large batch size and the effectiveness of the method to do so on a small GPU.

## Takeaways

This paper uses a simple property of the chain rule to separate the computation of the InfoNCE loss gradients into two parts, thereby removing the memory bottleneck. This method is implemented as the `CachedMultipleNegativesRankingLoss` in `SentenceTransformer`, and is really useful for those of us without access to large memory GPUs. We can train a model *exactly* to the performance of an arbitrarily large batch size, at the cost of longer computation time.

## Implementation

The author Luyu Gao has an implementation of GradCache in pytorch. The source code of the main classes are here:
- [PLGradCache](https://github.com/luyug/GradCache/blob/main/src/grad_cache/pytorch_lightning/pl_gradcache.py)
- [GradCache](https://github.com/luyug/GradCache/blob/main/src/grad_cache/grad_cache.py#L16)
- [RandContext](https://github.com/luyug/GradCache/blob/main/src/grad_cache/context_managers.py)

The main method is in `cache_step` which computes the loss for a mini batch. We follow the logic below:
- <<Step 1>>: `forward_no_grad` is called on the `model_inputs` to get the encoded representations (or embeddings) of all the input texts:
    - `torch.no_grad` is used as context manager to avoid gradients
    - In a for loop over the sub-batches:
        - The model `forward` method is called on the sub-batch input tensors
        - `RandContext` context manager for this forward pass is initialized and stored in a list of `rnd_states`
            - We need to store the random state of both CPUs and GPUs for this forward pass, because we need to exactly replicate the random number state at this point in time later. These random states can affect the behaviour of certain nn layers, especially DropOut. 
            - The `RandContext` object will be used as context manager later on
    - The sub-batch representation tensors are concatenated together  
    - The `model_reps` and `rnd_states` are returned
        - `model_reps` is appended to a list `all_model_reps`
        - `rnd_states` is appended to a list `all_rnd_states`
- <<Step 2>>: `build_cache` is called to build the cache of gradients. These are the gradients from the loss $\L$ to the embeddings $f(s_i), g(t_j)$.
    - `compute_loss` is called to forward pass from the embeddings to the loss
    - `backward` is called to compute the gradients from the loss to the embeddings
    - For each embedding `r`, `r.grad` is accessed to get the gradients
    - The cache is thus `[r.grad for r in reps]`
- <<Step 3>>: `forward_backward` is called to accumulate gradients 
    - Firstly, `with state` is called to restore the random context that we stored earlier. This ensures that the forward pass of the model to get the embedding (this time with gradients) exactly matches the earlier forward pass with no gradients.
    - We obtain the embeddings `y` with gradients enabled. This corresponds to $f(s_i)$ in the analysis above. 
    - We retrieve the gradients associated with each embedding that we stored earlier in step 2 (call it `reps`). This corresponds to $u_i$ in the analysis above.
    - Now we dot product `reps` and `y` to form `surrogate`. `backward` is then called on `surrogate` to get the correct gradients.
        - This step is a bit tricky, let's look at it in a bit more detail. 
        - Recall that our objective is to obtain $\frac{\partial L}{\partial f(s_i)} \frac{\partial f(s_i)}{\Theta}$
        - Recall that $u_i$ is the precomputed numerical value of $\frac{\partial L}{\partial f(s_i)}$
        - Since $u_i$ is a constant, it will become a constant multiplier to the gradient on the backward pass
        - Hence by calling backward on `surrogate`, we will get gradients of the form $u_i \cdot \frac{\partial f(s_i)}{\partial \Theta}$, which is what we want
    - After the `forward_backward` function, the gradients will be accumulated in `model.parameters().grad`, and the optimizer step can then be taken

`RandContext` itself is an interesting context manager to store and load pytorch's internal rng state.
- On `init`, the current cpu and gpu rng states are <<captured>>:
    - `torch.get_rng_state()` gets a byte tensor representing the cpu rng state
    - `torch.utils.checkpoint.get_device_states(*tensors)` looks up the gpu device where the tensors are held, and returns both the `gpu_devices` and `gpu_rng_states` across all gpu devices
- `__enter__` is triggered when this class is used as context manager, to <<restore>> the earlier captured rng state.
    - `self._fork = torch.random.fork_rng` is called to create a fork of the current torch rng state. This creates an isolated rng environment where we can restore the earlier rng environment without messing up the original rng environment that we entered from.
    - `self._fork.__enter__()` is called to actually enter the forked state
    - `torch.set_rng_state` now sets the cpu rng state to the earlier captured state
    - `torch.utils.checkpoint.set_device_states` similarly sets the gpu rng state to the earlier captured state
- `__exit__` is triggered when the context manager is closed.
    - `self._fork.__exit__` is called to close the isolated rng environment

