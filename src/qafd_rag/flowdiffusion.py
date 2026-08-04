from typing import Generic
from .graphinterface import NodeID, GraphProtocol
from .activenodes import ActiveSet
import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
            return -1 # DNE in this case
        cos_sim = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
        return (cos_sim + 1.0) / 2.0

class FlowDiffusion(Generic[NodeID]):
    def __init__(self, graph: GraphProtocol[NodeID], sources: list[NodeID] | dict[NodeID, float],
     query_embedding: list[float], confidence: float, epsilon: float,
     step_size: float, weight_func: str | None):

        self.mass : dict[NodeID, float] = {}
        self.sink_capacity : dict[NodeID, float] = {}
        self.active : ActiveSet[NodeID] = ActiveSet()
        self.graph = graph
        self.sources = sources
        self.query_embedding = query_embedding
        self.confidence = confidence
        self.epsilon = epsilon
        self.step_size = step_size
        self.weight_func = (weight_func or "").lower()
        self.edge_weights_cache : dict[tuple(NodeID, NodeID), float] = {}
        self.x: dict[NodeID, float] = {}

    def get_edge_weight(self, node1: NodeID, node2: NodeID) -> float:
        nodes = (node1, node2)
        if nodes in self.edge_weights_cache:
            return self.edge_weights_cache[nodes]

        static_weight = self.graph.neighbors(node1)[node2]

        cos_sim1e = cosine_similarity(self.graph.embedding(node1), self.query_embedding)
        cos_sim2e = cosine_similarity(self.graph.embedding(node2), self.query_embedding)

        if self.weight_func.lower() == "product":
            edge_weight = static_weight  * cos_sim1e * cos_sim2e
        elif self.weight_func.lower() == "avg":
            edge_weight = (static_weight + cos_sim1e + cos_sim2e) / 3.0
        else:
            edge_weight = static_weight * (1.0 + ((cos_sim1e + cos_sim2e) / 2.0) * 0.5)
        
        self.edge_weights_cache[nodes] = edge_weight
        return edge_weight

    def initialize(self, alpha: float=50):
        for node in self.graph.nodes():
            weighted_degree = sum(self.graph.neighbors(node).values())
            self.sink_capacity[node] = max(1.0, weighted_degree)

        for node in self.graph.nodes():
            self.mass[node] = 0.0
            self.x[node] = 0.0

        boosted_confidence = 1.0 + self.confidence

        # source_shares: node -> fraction of total seed mass it gets. Uniform
        # split when sources is a plain list (unchanged behavior); proportional
        # to caller-supplied weight when sources is a dict (e.g. an externally
        # fused dense+sparse score).
        if isinstance(self.sources, dict):
            total_weight = sum(self.sources.values()) or 1.0
            source_shares = {s: w / total_weight for s, w in self.sources.items()}
        else:
            source_shares = {s: 1.0 / len(self.sources) for s in self.sources}

        for source, share in source_shares.items():
            self.mass[source] = alpha * boosted_confidence * share

        for source, share in source_shares.items():
            self.x[source] = share

        for i in range(2):
            x_new = {}
            for node, val in self.x.items():
                if val > 0:
                    neighbors = list(self.graph.neighbors(node))
                    if neighbors:
                        for neighbor in neighbors:
                            x_new[neighbor] = x_new.get(neighbor, 0.0) + val / len(neighbors)

            x_combined = {node: 0.0 for node in self.graph.nodes()}
            for source, share in source_shares.items():
                x_combined[source] = share

            for node in set(x_new) | set(source_shares):
                x_combined[node] = (x_combined.get(node, 0.0) + x_new.get(node, 0.0)) / 2.0

            self.x = x_combined

    def push(self, node: str) -> bool:
        neighbor_weights = self.graph.neighbors(node)
        if not neighbor_weights: return False
        w_aq = 0
        for neighbor_weight in neighbor_weights:
            w_aq += self.get_edge_weight(node, neighbor_weight)
        if w_aq == 0: return False
        excess = self.mass[node] - self.sink_capacity[node]
        if excess <= 0: return False
        w_struct = sum(neighbor_weights.values())
        self.x[node] += self.step_size * excess / (w_struct + 1e-8)
        self.mass[node] = self.sink_capacity[node]
        for neighbor_weight in neighbor_weights:
            self.mass[neighbor_weight] += excess * self.get_edge_weight(node, neighbor_weight) / (w_aq + 1e-8)
        return True

    def flow_diffusion(self, max_iters: int=500) -> dict[NodeID, float]:
        for source in self.sources:
            excess = self.mass[source] - self.sink_capacity[source]
            if excess > 0:
                self.active.add(source)
        
        total_excess = sum((max(0.0, self.mass[vertex] - self.sink_capacity[vertex]) for vertex in self.graph.nodes()))
        iter = 0
        while total_excess > self.epsilon and self.active.has_active():
            node = self.active.pick_random()
            success = self.push(node)
            if success:
                self.active.remove(node)
                for neighbor in self.graph.neighbors(node):
                    if (self.mass[neighbor] > self.sink_capacity[neighbor]) and not self.active.is_active(neighbor):
                        self.active.add(neighbor)
            else:
                self.active.remove(node)
            iter += 1
            if (iter >= max_iters): 
                print("Does not converge before reaching max iters")
                break
            total_excess = sum((max(0.0, self.mass[vertex] - self.sink_capacity[vertex]) for vertex in self.graph.nodes()))
        
        return {node: val for node, val in self.x.items() if val > 0}    