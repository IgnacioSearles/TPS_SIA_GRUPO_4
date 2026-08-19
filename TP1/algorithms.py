from state import State, Level
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod

class SearchAlgorithm(ABC):
    @abstractmethod
    def solve(self, initial_state: State) -> Optional[Tuple[State, int, int]]:
        """
        Executes the search.
        Returns: (final_goal_state, expanded_nodes_count, frontier_nodes_count)
        Returns None if no solution is found.
        """
        pass

class BFS(SearchAlgorithm):
    def solve(self, initial_state: State):
        pass

class AStar(SearchAlgorithm):
    def __init__(self, heuristic: Callable[[State, Level], float]):
        self.heuristic = heuristic

    def solve(self, initial_state: State):
        pass