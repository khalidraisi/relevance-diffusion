# qafd-rag

A backend-agnostic Python implementation of **Query-Aware Flow Diffusion (QAFD)** for
graph-based retrieval ([paper](https://openreview.net/forum?id=n28wnc2QTc)).

Instead of ranking by flat vector similarity, it treats retrieval as a flow-diffusion
process over a graph: mass is seeded on query-relevant nodes and diffused across edges
weighted by both graph structure and query similarity, surfacing context that is
*structurally* connected to the query. It works over any graph store (NetworkX, Neo4j, a
custom adjacency dict, …) through a small three-method interface.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10 and `numpy`.

## Usage

Implement the graph interface, then call `retrieve()`:

```python
from qafd_rag import retrieve

class MyGraph:
    def nodes(self):            # -> Iterable[NodeID]
        ...
    def neighbors(self, node):  # -> dict[NodeID, float]   (neighbor -> structural weight)
        ...
    def embedding(self, node):  # -> list[float]
        ...

results = retrieve(
    graph=MyGraph(),
    query="how do cats regulate body temperature?",
    keyword_extractor=lambda q: q.split(),   # your keyword extractor
    embed_fn=my_embedding_model,             # str -> list[float]
    n_seeds=5,
    confidence=0.5,
    epsilon=1e-3,
    step_size=0.5,
    weight_func=None,   # None (default blend) | "product" | "avg"
)
# results: list[(node_id, score)] sorted by relevance, highest first
top_nodes = [node for node, _ in results]
```

To bypass seed selection and pass pre-scored seeds directly (e.g. a fused dense+sparse
score), use `seed_weights={node_id: weight, ...}`.

## License

MIT
