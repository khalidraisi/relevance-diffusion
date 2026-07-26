from typing import Generic
from .graphinterface import NodeID, GraphProtocol
from .activenodes import ActiveSet
import numpy as np

class FlowDiffusion(Generic[NodeID]):
    def __init__(self, graph: GraphProtocol[NodeID], source: NodeID,
     query_embedding: list[float], confidence: float, epsilon: float,
     step_size: float, weight_func: str | None):

        self.mass : dict[NodeID, float] = {}
        self.sink_capacity : dict[NodeID, float] = {}
        self.active : ActiveSet[NodeID] = ActiveSet()
        self.graph = graph
        self.source = source
        self.query_embedding = query_embedding
        self.confidence = confidence
        self.epsilon = epsilon
        self.step_size = step_size
        self.weight_func = weight_func
        self.edge_weights_cache : dict[NodeID, float] = {}

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
            return -1 # DNE in this case
        cos_sim = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
        return (cos_sim + 1.0) / 2.0