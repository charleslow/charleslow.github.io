# Hamilton 2017 - GraphSAGE

[Paper](https://ar5iv.labs.arxiv.org/html/1706.02216).

This paper presents a framework to efficiently generate node embeddings, especially for previously unseen nodes. It calls this <<inductive>> learning, i.e. the model is able to generalize to new nodes, as opposed to previous frameworks which are <<transductive>> and only learn embeddings for seen nodes. For example, matrix factorization methods are transductive because we can only make predictions on a graph with fixed nodes, and need to be retrained when new nodes are added.

GraphSAGE (i.e. Sample and Aggregation) aggregates feature information from the neighbourhood of a node to represent a given node. Feature information can include structural information (e.g. degree of a node) or other content-based information about a node, e.g. description of a job for a job node etc. 

## Setup

We start with a graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$ that is provided by data. We have input features $x_v \forall v \in \mathcal{V}$. For a user-chosen depth of $K$, we have $K$ aggregator functions $\texttt{AGG}_k$ and $K$ weight matrices $W_k$. We also have a neighbourhood function $\mathcal{N}: \mathcal{V} \rightarrow 2^\mathcal{V}$, which means it maps from a node $v \in \mathcal{V}$ to a set of nodes in $\mathcal{V}$. Note that $2^\mathcal{V}$ is the powerset of the set $\mathcal{V}$, i.e. the set of all possible subsets of $\mathcal{V}$. We wish to generate low-dimensional, dense vector representations of each node $z_v$.

## Forward Propagation

The algorithm for the forward propagation (in words) is as follows:
1. We start with hidden representations $h_v^0 \leftarrow x_v \forall v \in \mathcal{V}$, i.e. at layer 0, we just use the input features to represent each node
2. At depth $k=1$, we perform a neighbourhood aggregation step at each node $v$: 
$$h^k_{\mathcal{N}(v)} \leftarrow \texttt{AGG}_k(\{ h^{k-1}_u: u \in \mathcal{N}(v) \})$$
3. The aggregated vector is then passed through a dense layer to get the hidden representation at depth $k$. Note that $\sigma$ represents a non-linear activation, such as `ReLU`: 
$$h^k_v = \sigma \left( W_k \cdot \texttt{CONCAT}(h^{k-1}_v, h^k_{\mathcal{N}(v)}) \right)$$
4. We L2-normalize each vector $h^k_v \ \forall \ v \in \mathcal{V}$
5. We then repeat this process repeatedly for depths $k= 1,...,K$
6. We then take the last layer: $z_v \leftarrow h^k_v$

The intuition behind the forward propagation is that we use the neighbours of $v$ to represent it. Importantly, we also include the hidden representation of the current node from the previous depth (analogous to a residual connection). At each depth level $k$, we increasingly pull more information from further reaches of the graph. Note that in the aggregation step, a subset of each node's neighbours are sampled uniformly (as opposed to taking the full neighbour set) to control the complexity of the algorithm.

## Loss

To train the weights, we define an <<unsupervised loss>> based on how well the embeddings are able to reconstruct the graph. Specifically, we have a loss which:
- Rewards positive pairs for having a high dot product
- Penalizes negative pairs ($v_n$ being sampled negatives according to the negative sampling distribution $P_n$)

$$
    J_{\mathcal{G}}(z_u) = 
        -log(\sigma(z_u^Tz_v)) - 
        Q \cdot \mathbf{E}_{v_n \sim P_n(v)} log(\sigma(-z_u^T z_{v_n}))
$$

Alternatively, we can also define a <<supervised loss>> based on classification cross entropy loss, with presumably some form of negative sampling. The authors did not elaborate on this.

## Aggregation Methods

The authors explored a few ways to define the $\texttt{AGG}$ function to aggregate neighbour embeddings together:
- `GraphSAGE-mean`: The element-wise mean of the neighbour embeddings is taken
- `GraphSAGE-GCN`: Same as above, except that the current node's hidden representation from the previous depth $h_v^{k-1}$ is not included. The experiments show that omitting this residual connection actually leads to significant performance degradation.
- `GraphSAGE-LSTM`: An LSTM is fitted over the sequence of embeddings. Since there is no inherent order to the neighbours, the authors randomize the ordering for each training sample
- `GraphSAGE-pool`: An additional linear layer is added over the sequence of embeddings, before an element-wise `max-pool` operation is carried out

Generally from the experiments, it seems that `GraphSAGE-mean` is sufficient.