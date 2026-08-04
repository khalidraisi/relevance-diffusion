from typing import Callable
from .graphinterface import NodeID, GraphProtocol
from .seedselection import select_seeds, KeywordExtractor
from .flowdiffusion import FlowDiffusion

def retrieve(
    graph: GraphProtocol[NodeID], query: str, keyword_extractor: KeywordExtractor,
    embed_fn: Callable[[str], list[float]], n_seeds: int, confidence: float,
    epsilon: float, step_size: float, weight_func: str | None, alpha: float = 50,
    max_iters: int = 500, seed_weights: dict[NodeID, float] | None = None,
    ) -> list[tuple[NodeID, float]]:

    seeds = seed_weights if seed_weights is not None else select_seeds(graph, query, keyword_extractor, embed_fn, n_seeds)
    query_embedding = embed_fn(query)
    flowdiffusion = FlowDiffusion(graph, seeds, query_embedding, confidence, epsilon, step_size, weight_func)
    flowdiffusion.initialize(alpha)

    return sorted(flowdiffusion.flow_diffusion(max_iters).items(), key=lambda pair: pair[1], reverse=True)