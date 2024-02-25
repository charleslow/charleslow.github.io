# Hamilton 2017 - GraphSAGE

[Paper](https://ar5iv.labs.arxiv.org/html/1706.02216).

This paper presents a framework to efficiently generate node embeddings, especially for previously unseen nodes. It calls this <inductive> learning, i.e. the model is able to generalize to new nodes, as opposed to previous frameworks which are <transductive> and only learn embeddings for seen nodes. For example, matrix factorization methods are transductive because we can only make predictions on a graph with fixed nodes, and need to be retrained when new nodes are added.

GraphSAGE (i.e. Sample and Aggregation) aggregates feature information from the neighbourhood of a node to represent a given node. Feature information can include structural information (e.g. degree of a node) or other content-based information about a node, e.g. description of a job for a job node etc. 

