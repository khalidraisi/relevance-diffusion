from typing import Callable
from .graphinterface import NodeID, GraphProtocol
from .flowdiffusion import cosine_similarity

KeywordExtractor = Callable[[str], list[str]]

def select_seeds(
    graph: GraphProtocol[NodeID], query: str, keyword_extractor: KeywordExtractor,
    embed_fn: Callable[[str], list[float]], n: int,) -> list[NodeID]:
    keywords = keyword_extractor(query)

    kw_embs: list[list[float]] = []
    for keyword in keywords: kw_embs.append(embed_fn(keyword))

    scores = []
    for node in graph.nodes():
        node_emb = graph.embedding(node)
        best = max(cosine_similarity(node_emb, kw_emb) for kw_emb in kw_embs)
        scores.append((node, best))

    scores = sorted(scores, key=lambda pair: pair[1], reverse=True)[:n]
    return [node for node, _ in scores]