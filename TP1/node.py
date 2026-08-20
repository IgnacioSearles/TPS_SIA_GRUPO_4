from dataclasses import dataclass
from typing import Optional, List

from state import Action, State


@dataclass
class SearchNode:
    state: State
    parent: Optional['SearchNode']
    action: Optional[Action]
    cost: int
    depth: int

    @staticmethod
    def root(state: State) -> 'SearchNode':
        return SearchNode(
            state=state,
            parent=None,
            action=None,
            cost=0,
            depth=0,
        )

    def expand_child(self, state: State, action: Action, step_cost: int) -> 'SearchNode':
        return SearchNode(
            state=state,
            parent=self,
            action=action,
            cost=self.cost + step_cost,
            depth=self.depth + 1,
        )

    def reconstruct_path(self) -> List[Action]:
        path = []
        current: Optional[SearchNode] = self
        while current is not None and current.action is not None:
            path.append(current.action)
            current = current.parent
        path.reverse()
        return path
