from algorithms import SearchAlgorithm
from state import State, Action, Level
from node import SearchNode
from typing import List
from dataclasses import dataclass
import time

@dataclass
class SearchResult:
    success: bool
    path: List[Action]
    cost: int
    expanded_nodes: int
    # Maximum number of pending nodes held by the search frontier.
    frontier_nodes: int
    processing_time_sec: float

def run_search(algorithm: SearchAlgorithm, initial_state: State, level: Level) -> SearchResult:
    start_time = time.perf_counter()

    result = algorithm.solve(initial_state, level)

    processing_time = time.perf_counter() - start_time

    if result is None:
        return SearchResult(
            success=False,
            path=[],
            cost=0,
            expanded_nodes=0,
            frontier_nodes=0,
            processing_time_sec=processing_time
        )

    final_node, expanded_count, frontier_count = result
    path = reconstruct_path(final_node)

    return SearchResult(
        success=True,
        path=path,
        cost=final_node.cost,
        expanded_nodes=expanded_count,
        frontier_nodes=frontier_count,
        processing_time_sec=processing_time
    )

def reconstruct_path(final_node: SearchNode) -> List[Action]:
    return final_node.reconstruct_path()
