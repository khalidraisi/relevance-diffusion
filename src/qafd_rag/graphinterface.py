from typing import Protocol, Hashable, Iterable, TypeVar, Generic, runtime_checkable

NodeID = TypeVar("NodeID", bound=Hashable)

@runtime_checkable
class GraphProtocol(Protocol, Generic[NodeID]):
    def nodes(self) -> Iterable[NodeID]:
        raise NotImplementedError
    def neighbors(self, node: NodeID) -> dict[NodeID, float]:
        raise NotImplementedError
    def embedding(self, node: NodeID) -> list[float]:
        raise NotImplementedError