from typing import Generic
from .graphinterface import NodeID
import random

class ActiveSet(Generic[NodeID]):
    def __init__(self):
        self._items: list[NodeID]=[]
        self._index: dict[NodeID, int] = {}

    def add(self, node: NodeID):
        self._items.append(node)
        self._index[node] = len(self._items) - 1

    def remove(self, node: NodeID):
        node_index = self._index[node]
        last_element = self._items[-1]
        self._items[node_index] = last_element
        self._index[last_element] = node_index
        self._items.pop()
        self._index.pop(node)

    def has_active(self) -> bool:
        return len(self._items) > 0

    def pick_random(self) -> NodeID:
        if not self.has_active():
            return None
        return random.choice(self._items)


