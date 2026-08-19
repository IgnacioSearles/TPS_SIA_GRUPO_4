from state import State, Level
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod
from collections import deque

class SearchAlgorithm(ABC):
    @abstractmethod
    def solve(self, initial_state: State, level: Level) -> Optional[Tuple[State, int, int]]:
        """
        Executes the search.
        Returns: (final_goal_state, expanded_nodes_count, frontier_nodes_count)
        Returns None if no solution is found.
        """
        pass

class BFS(SearchAlgorithm):
    def solve(self, initial_state: State, level: Level) -> Optional[Tuple[State, int, int]]:
        frontier = deque([initial_state])

        visited = set()
        visited.add(initial_state)

        expanded_nodes = 0

        while frontier:
            current_state = frontier.popleft()

            if current_state.is_goal(level):
                return current_state, expanded_nodes, len(frontier)

            expanded_nodes += 1
            for succesor in current_state.get_successors(level):
                if succesor not in visited:
                    visited.add(succesor)
                    frontier.append(succesor)

        return None


class AStar(SearchAlgorithm):
    def __init__(self, heuristic: Callable[[State, Level], float]):
        self.heuristic = heuristic

    def solve(self, initial_state: State, level: Level) -> Optional[Tuple[State, int, int]]:
        pass