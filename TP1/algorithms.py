from state import PRUNING_MODES, Level, State
from typing import Callable, Optional, Tuple
from abc import ABC, abstractmethod
from collections import deque
import heapq
from itertools import count

from heuristics import (
    boxes_to_goals,
    boxes_to_goals_and_player_all,
    boxes_to_goals_and_player_unsolved,
    boxes_to_goals_hungarian,
    boxes_to_goals_hungarian_and_player_all,
    boxes_to_goals_hungarian_and_player_unsolved,
    boxes_to_goals_push_distance,
    boxes_to_goals_push_distance_and_player_all,
    boxes_to_goals_push_distance_and_player_unsolved,
    boxes_to_goals_push_distance_nearest,
    boxes_to_goals_push_distance_nearest_and_player,
)

HEURISTICS = {
    "manhattan": boxes_to_goals,
    "manhattan_player": boxes_to_goals_and_player_unsolved,
    "manhattan_player_all": boxes_to_goals_and_player_all,
    "manhattan_player_unsolved": boxes_to_goals_and_player_unsolved,
    "hungarian": boxes_to_goals_hungarian,
    "hungarian_player": boxes_to_goals_hungarian_and_player_unsolved,
    "hungarian_player_all": boxes_to_goals_hungarian_and_player_all,
    "hungarian_player_unsolved": boxes_to_goals_hungarian_and_player_unsolved,
    "push_distance": boxes_to_goals_push_distance,
    "push_distance_player": boxes_to_goals_push_distance_and_player_unsolved,
    "push_distance_player_all": boxes_to_goals_push_distance_and_player_all,
    "push_distance_player_unsolved": boxes_to_goals_push_distance_and_player_unsolved,
    "push_distance_nearest": boxes_to_goals_push_distance_nearest,
    "push_distance_nearest_player": boxes_to_goals_push_distance_nearest_and_player,
}

class SearchAlgorithm(ABC):
    def __init__(self, pruning_mode: str = "dead_squares"):
        if pruning_mode not in PRUNING_MODES:
            valid_modes = ", ".join(sorted(PRUNING_MODES))
            raise ValueError(
                f"Unknown pruning mode '{pruning_mode}'. Valid options: {valid_modes}"
            )
        self.pruning_mode = pruning_mode

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
            for successor in current_state.get_successors(level, self.pruning_mode):
                if successor not in visited:
                    visited.add(successor)
                    frontier.append(successor)

        return None


class DFS(SearchAlgorithm):
    def solve(self, initial_state: State, level: Level) -> Optional[Tuple[State, int, int]]:
        frontier = [initial_state]

        visited = set()
        visited.add(initial_state)

        expanded_nodes = 0

        while frontier:
            current_state = frontier.pop()

            if current_state.is_goal(level):
                return current_state, expanded_nodes, len(frontier)

            expanded_nodes += 1
            for successor in current_state.get_successors(level, self.pruning_mode):
                if successor not in visited:
                    visited.add(successor)
                    frontier.append(successor)

        return None


class AStar(SearchAlgorithm):
    def __init__(
        self,
        heuristic: Callable[[State, Level], float] = boxes_to_goals_hungarian,
        pruning_mode: str = "dead_squares",
    ):
        super().__init__(pruning_mode)
        self.heuristic = heuristic

    def solve(self, initial_state: State, level: Level) -> Optional[Tuple[State, int, int]]:
        tie_breaker = count()
        frontier = []
        initial_heuristic = self.heuristic(initial_state, level)
        initial_priority = initial_state.cost + initial_heuristic
        heapq.heappush(
            frontier,
            (
                initial_priority,
                initial_heuristic,
                next(tie_breaker),
                initial_state,
            ),
        )

        best_cost = {initial_state: initial_state.cost}
        expanded_nodes = 0

        while frontier:
            _, _, _, current_state = heapq.heappop(frontier)

            # Ignore an obsolete entry superseded by a cheaper path.
            if current_state.cost != best_cost.get(current_state):
                continue

            if current_state.is_goal(level):
                return current_state, expanded_nodes, len(frontier)

            expanded_nodes += 1
            for successor in current_state.get_successors(level, self.pruning_mode):
                successor_cost = successor.cost
                if successor_cost >= best_cost.get(successor, float("inf")):
                    continue

                best_cost[successor] = successor_cost
                successor_heuristic = self.heuristic(successor, level)
                priority = successor_cost + successor_heuristic
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        successor_heuristic,
                        next(tie_breaker),
                        successor,
                    ),
                )

        return None


class Greedy(SearchAlgorithm):
    def __init__(
        self,
        heuristic: Callable[[State, Level], float] = boxes_to_goals_hungarian,
        pruning_mode: str = "dead_squares",
    ):
        super().__init__(pruning_mode)
        self.heuristic = heuristic

    def solve(self, initial_state: State, level: Level) -> Optional[Tuple[State, int, int]]:
        tie_breaker = count()
        frontier = []
        heapq.heappush(
            frontier,
            (self.heuristic(initial_state, level), next(tie_breaker), initial_state),
        )

        visited = set()
        visited.add(initial_state)
        expanded_nodes = 0

        while frontier:
            _, _, current_state = heapq.heappop(frontier)

            if current_state.is_goal(level):
                return current_state, expanded_nodes, len(frontier)

            expanded_nodes += 1
            for successor in current_state.get_successors(level, self.pruning_mode):
                if successor not in visited:
                    visited.add(successor)
                    priority = self.heuristic(successor, level)
                    heapq.heappush(frontier, (priority, next(tie_breaker), successor))

        return None


class IDDFS(SearchAlgorithm):
    def __init__(
        self,
        max_depth: Optional[int] = None,
        pruning_mode: str = "dead_squares",
    ):
        super().__init__(pruning_mode)
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        self.max_depth = max_depth

    def solve(self, initial_state: State, level: Level) -> Optional[Tuple[State, int, int]]:
        depth_limit = 0
        expanded_nodes = 0

        while self.max_depth is None or depth_limit <= self.max_depth:
            result = self._depth_limited_search(
                initial_state,
                level,
                depth_limit,
            )
            if result is not None:
                final_state, iteration_expanded, frontier_nodes = result
                return final_state, expanded_nodes + iteration_expanded, frontier_nodes

            expanded_nodes += self._last_iteration_expanded
            depth_limit += 1

        return None

    def _depth_limited_search(
        self,
        initial_state: State,
        level: Level,
        depth_limit: int,
    ) -> Optional[Tuple[State, int, int]]:
        expanded_nodes = 0
        frontier = [(initial_state, 0)]
        visited = {initial_state: 0}

        while frontier:
            current_state, depth = frontier.pop()

            # Ignore an older occurrence of the same state reached deeper.
            if depth != visited[current_state]:
                continue

            if current_state.is_goal(level):
                self._last_iteration_expanded = expanded_nodes
                return current_state, expanded_nodes, len(frontier)

            if depth == depth_limit:
                continue

            expanded_nodes += 1
            for successor in current_state.get_successors(level, self.pruning_mode):
                successor_depth = depth + 1
                if successor_depth >= visited.get(successor, float("inf")):
                    continue

                visited[successor] = successor_depth
                frontier.append((successor, successor_depth))

        self._last_iteration_expanded = expanded_nodes
        return None

ALGORITHMS = {
    "bfs": BFS,
    "dfs": DFS,
    "astar": AStar,
    "greedy": Greedy,
    "iddfs": IDDFS,
}
