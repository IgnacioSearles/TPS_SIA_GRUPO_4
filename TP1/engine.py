from algorithms import SearchAlgorithm
from state import State, Action, Level
from typing import List
from dataclasses import dataclass
import time

@dataclass
class SearchResult:
    success: bool
    path: List[Action]
    cost: int
    expanded_nodes: int
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

    final_state, expanded_count, frontier_count = result
    path = reconstruct_path(final_state)

    return SearchResult(
        success=True,
        path=path,
        cost=final_state.cost,
        expanded_nodes=expanded_count,
        frontier_nodes=frontier_count,
        processing_time_sec=processing_time
    )

def reconstruct_path(final_state: State) -> List[Action]:
    path = []
    current = final_state
    while current.parent is not None:
        if current.action:
            path.append(current.action)
        current = current.parent
    path.reverse()
    return path