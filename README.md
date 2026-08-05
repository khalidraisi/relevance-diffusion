# qafd-rag

A from-scratch implementation of **Query-Aware Flow Diffusion (QAFD)** for graph-based
retrieval — see the [paper](https://openreview.net/forum?id=n28wnc2QTc).

Instead of ranking documents by flat vector similarity, QAFD treats retrieval as a
**flow-diffusion process over a graph**: seed mass is placed on query-relevant nodes and
diffused across edges whose weights combine graph structure with query-aware embedding
similarity. This surfaces context that is *structurally* connected to the query, not just
lexically or semantically close in isolation.

The library is **backend-agnostic** — it talks to your graph through a small `Protocol`,
so it works over any store (NetworkX, Neo4j, a custom adjacency dict, …) without changes.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10 and `numpy`.

## Usage

Implement the three-method graph interface, then call `retrieve()`:

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
    keyword_extractor=lambda q: q.split(),      # your keyword extractor
    embed_fn=my_embedding_model,                # str -> list[float]
    n_seeds=5,
    confidence=0.5,
    epsilon=1e-3,
    step_size=0.5,
    weight_func=None,   # None (default blend) | "product" | "avg"
)
# results: list[(node_id, score)] sorted by relevance, highest first
top_nodes = [node for node, _ in results]
```

You can also bypass seed selection and pass pre-scored seeds directly (e.g. a fused
dense+sparse retrieval score) via `seed_weights={node_id: weight, ...}`.

## Results

Used as the retriever in a production RAG pipeline during an internship and evaluated with
[DeepEval](https://github.com/confident-ai/deepeval). The pipeline achieved **~90%
contextual precision, contextual recall, and faithfulness at top-k = 5** — strong accuracy
at a *low* k, indicating the retriever returns tightly-scoped, low-noise context rather
than padding recall with a large candidate set.

> Metrics reflect the end-to-end RAG pipeline this retriever powered, not the library in
> isolation.

## How it works

1. **Seed selection** (`seedselection.py`) — extract keywords from the query, embed them,
   and pick the top-`n` nodes by embedding similarity as diffusion seeds.
2. **Query-aware edge weights** (`flowdiffusion.py`) — each edge weight blends its
   structural weight with the cosine similarity of both endpoints to the query embedding
   (`product` / `avg` / default blend).
3. **Flow diffusion** — seed mass is placed on seeds and pushed to neighbors whenever a
   node's mass exceeds its sink capacity (a local-push scheme), accumulating a relevance
   score `x` per node.
4. **Rank** — return nodes with positive score, sorted descending.

## API

- `retrieve(graph, query, keyword_extractor, embed_fn, n_seeds, confidence, epsilon, step_size, weight_func, alpha=50, max_iters=500, seed_weights=None)`
- `FlowDiffusion` — the diffusion solver, if you want lower-level control.
- `select_seeds(...)` — standalone seed selection.
- `GraphProtocol` — the interface your graph must satisfy.

## License

MIT
