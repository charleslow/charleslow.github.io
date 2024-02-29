# Hamilton 2017 - GraphSAGE

[Paper](https://ar5iv.labs.arxiv.org/html/1706.02216).

This paper presents a framework to efficiently generate node embeddings, especially for previously unseen nodes. It calls this <inductive> learning, i.e. the model is able to generalize to new nodes, as opposed to previous frameworks which are <transductive> and only learn embeddings for seen nodes. For example, matrix factorization methods are transductive because we can only make predictions on a graph with fixed nodes, and need to be retrained when new nodes are added.

GraphSAGE (i.e. Sample and Aggregation) aggregates feature information from the neighbourhood of a node to represent a given node. Feature information can include structural information (e.g. degree of a node) or other content-based information about a node, e.g. description of a job for a job node etc. 

## Setup

We start with a graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$ that is provided by data. We have input features $x_v \forall v \in \mathcal{V}$. For a user-chosen depth of $K$, we have $K$ aggregator functions $\texttt{AGG}_k$ and $K$ weight matrices $W_k$. We also have a neighbourhood function $\mathcal{N}: \mathcal{V} \rightarrow 2^\mathcal{V}$, which means it maps from a node $v \in \mathcal{V}$ to a set of nodes in $\mathcal{V}$. Note that $2^\mathcal{V}$ is the powerset of the set $\mathcal{V}$, i.e. the set of all possible subsets of $\mathcal{V}$. We wish to generate low-dimensional, dense vector representations of each node $z_v$.

## Forward Propagation

The algorithm for the forward propagation (in words) is as follows:
1. We start with hidden representations $h_v^0 \leftarrow x_v \forall v \in \mathcal{V}$, i.e. at layer 0, we just use the input features to represent each node
2. At depth $k=1$, we perform a neighbourhood aggregation step at each node $v$: 
$$h^k_{\mathcal{N}(v)} \leftarrow \texttt{AGG}_k(\{ h^{k-1}_u: u \in \mathcal{N}(v) \})$$
3. The aggregated vector is then passed through a dense layer to get the hidden representation at depth $k$: 
$$h^k_v = \sigma \left( W_k \cdot \texttt{CONCAT}(h^{k-1}_v, h^k_{\mathcal{N}(v)}) \right)$$
4. We L2-normalize each vector $h^k_v \forall v \in \mathcal{V}$
5. We then repeat this process repeatedly for depths $k= 1,...,K$
6. We then take the last layer: $z_v \leftarrow h^k_v$

The intuition behind the forward propagation is that we use the neighbours of $v$ to represent it. At each depth level $k$, we increasingly pull more information from further reaches of the graph. 