from collections import deque
from functools import lru_cache
from typing import FrozenSet

from state import State, Level

def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest_box_distance(
    state: State,
    level: Level,
    include_solved: bool,
) -> int:
    """Distance from the player to the nearest selected box."""
    boxes = state.boxes if include_solved else state.boxes - level.goals
    if not boxes:
        return 0

    return min(
        manhattan(state.player_pos, box)
        for box in boxes
    )


def _nearest_unsolved_box_distance(state: State, level: Level) -> int:
    return _nearest_box_distance(state, level, include_solved=False)


# ── Simple (non-exclusive) ────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _boxes_to_goals_cached(
    boxes: FrozenSet[tuple[int, int]],
    goals: FrozenSet[tuple[int, int]],
) -> int:
    """Cached box-only Manhattan cost for a fixed box layout."""
    return sum(
        min(manhattan(box, goal) for goal in goals)
        for box in boxes
    )


def boxes_to_goals(state: State, level: Level) -> int:
    """Sum of each box's distance to its nearest goal (goals can be shared)."""
    return _boxes_to_goals_cached(state.boxes, level.goals)

def boxes_to_goals_and_player(state: State, level: Level) -> int:
    """boxes_to_goals + player distance to nearest box."""
    return boxes_to_goals_and_player_unsolved(state, level)


def boxes_to_goals_and_player_all(state: State, level: Level) -> int:
    """Manhattan box cost plus distance to the nearest box, including solved boxes."""
    return (
        _boxes_to_goals_cached(state.boxes, level.goals)
        + _nearest_box_distance(state, level, include_solved=True)
    )


def boxes_to_goals_and_player_unsolved(state: State, level: Level) -> int:
    """Manhattan box cost plus distance to the nearest unsolved box."""
    return (
        _boxes_to_goals_cached(state.boxes, level.goals)
        + _nearest_unsolved_box_distance(state, level)
    )


# ── Wall-aware reverse push distances ────────────────────────────────────────

@lru_cache(maxsize=None)
def _reverse_push_distances(
    walls: FrozenSet[tuple[int, int]],
    goals: FrozenSet[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Compute the minimum pushes from every cell to any goal.

    The search runs backwards from the goals. A reverse transition is valid
    when a box could have been pushed from the predecessor cell to the current
    cell: both the predecessor and the player's support cell must be free.
    Other boxes and the player's actual position are intentionally ignored, so
    these distances remain a lower bound for A*.
    """
    known_cells = walls | goals
    min_x = min(x for x, _ in known_cells)
    max_x = max(x for x, _ in known_cells)
    min_y = min(y for _, y in known_cells)
    max_y = max(y for _, y in known_cells)

    floor = {
        (x, y)
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
        if (x, y) not in walls
    }

    distances = {goal: 0 for goal in goals}
    frontier = deque(goals)
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))

    while frontier:
        current_x, current_y = frontier.popleft()
        current_distance = distances[(current_x, current_y)]

        for dx, dy in directions:
            predecessor = (current_x - dx, current_y - dy)
            support = (current_x - 2 * dx, current_y - 2 * dy)

            if predecessor not in floor or support not in floor:
                continue
            if predecessor in distances:
                continue

            distances[predecessor] = current_distance + 1
            frontier.append(predecessor)

    return distances


def boxes_to_goals_push_distance(state: State, level: Level) -> int | float:
    """Minimum wall-aware box-to-goal assignment cost in pushes.

    Unlike Manhattan distance, this heuristic respects walls and impossible
    push directions. It ignores other boxes and the player's reachability, so
    it is still a lower bound on the number of moves required by a solution.
    """
    if not state.boxes:
        return 0

    return _boxes_to_goals_push_distance_cached(
        state.boxes,
        level.walls,
        level.goals,
    )


def boxes_to_goals_push_distance_and_player(state: State, level: Level) -> int | float:
    """Wall-aware push assignment cost plus distance to the nearest box."""
    return boxes_to_goals_push_distance_and_player_unsolved(state, level)


def boxes_to_goals_push_distance_and_player_all(
    state: State,
    level: Level,
) -> int | float:
    """Wall-aware push cost plus distance to any box, including solved boxes."""
    push_cost = _boxes_to_goals_push_distance_cached(
        state.boxes,
        level.walls,
        level.goals,
    )
    return push_cost + _nearest_box_distance(state, level, include_solved=True)


def boxes_to_goals_push_distance_and_player_unsolved(
    state: State,
    level: Level,
) -> int | float:
    """Wall-aware push cost plus distance to an unsolved box."""
    push_cost = _boxes_to_goals_push_distance_cached(
        state.boxes,
        level.walls,
        level.goals,
    )
    return push_cost + _nearest_unsolved_box_distance(state, level)


def boxes_to_goals_push_distance_nearest(
    state: State,
    level: Level,
) -> int | float:
    """Sum of each box's nearest wall-aware push distance to a goal.

    Unlike the Hungarian variants, goals may be selected by more than one box
    in this relaxed lower bound. This is cheaper to evaluate and remains
    admissible because it can only underestimate the real assignment cost.
    """
    return _boxes_to_goals_push_distance_nearest_cached(
        state.boxes,
        level.walls,
        level.goals,
    )


def boxes_to_goals_push_distance_nearest_and_player(
    state: State,
    level: Level,
) -> int | float:
    """Nearest wall-aware push cost plus distance to an unsolved box."""
    return (
        _boxes_to_goals_push_distance_nearest_cached(
            state.boxes,
            level.walls,
            level.goals,
        )
        + _nearest_unsolved_box_distance(state, level)
    )


@lru_cache(maxsize=None)
def _boxes_to_goals_push_distance_cached(
    boxes: FrozenSet[tuple[int, int]],
    walls: FrozenSet[tuple[int, int]],
    goals: FrozenSet[tuple[int, int]],
) -> int | float:
    """Cached wall-aware push assignment cost for a fixed box layout."""
    unreachable = 10**6
    goals = list(goals)
    boxes = list(boxes)
    cost_matrix = []
    for box in boxes:
        row = []
        for goal in goals:
            goal_distances = _reverse_push_distances(
                walls,
                frozenset({goal}),
            )
            row.append(goal_distances.get(box, unreachable))
        cost_matrix.append(row)

    assignment_cost = _hungarian(cost_matrix)
    if assignment_cost >= unreachable:
        return float("inf")
    return assignment_cost


@lru_cache(maxsize=None)
def _boxes_to_goals_push_distance_nearest_cached(
    boxes: FrozenSet[tuple[int, int]],
    walls: FrozenSet[tuple[int, int]],
    goals: FrozenSet[tuple[int, int]],
) -> int | float:
    """Cached nearest-goal push cost for a fixed box layout."""
    unreachable = 10**6
    goal_distances = [
        _reverse_push_distances(walls, frozenset({goal}))
        for goal in goals
    ]

    total_cost = 0
    for box in boxes:
        box_cost = min(
            distances.get(box, unreachable)
            for distances in goal_distances
        )
        if box_cost >= unreachable:
            return float("inf")
        total_cost += box_cost

    return total_cost


# ── Hungarian (optimal 1-to-1 assignment) ─────────────────────────────────────

def _hungarian(cost_matrix: list[list[int]]) -> int:
    """
    Returns the minimum total cost of a perfect 1-to-1 assignment.

    Operates in four repeating steps:
      1. Row reduction: subtract each row's minimum so every row has a zero.
      2. Col reduction: subtract each column's minimum so every col has a zero.
      3. Find a perfect matching on zero-cost cells via augmenting paths.
         If found, we're done — sum up the original costs for matched pairs.
      4. If not, find the smallest uncovered value, subtract it from all
         uncovered cells and add it to all doubly-covered cells, then retry.
    """
    n = len(cost_matrix)
    C = [row[:] for row in cost_matrix]

    # Step 1 – row reduction
    for i in range(n):
        m = min(C[i])
        for j in range(n):
            C[i][j] -= m

    # Step 2 – column reduction
    for j in range(n):
        m = min(C[i][j] for i in range(n))
        for i in range(n):
            C[i][j] -= m

    while True:
        match_row = [-1] * n  # match_row[i] = column assigned to row i
        match_col = [-1] * n  # match_col[j] = row assigned to column j

        # Greedy seed: assign the first zero found per row
        for i in range(n):
            for j in range(n):
                if C[i][j] == 0 and match_col[j] == -1:
                    match_row[i] = j
                    match_col[j] = i
                    break

        # Augment any unmatched rows via alternating paths
        for i in range(n):
            if match_row[i] != -1:
                continue
            visited_rows = [False] * n
            visited_cols = [False] * n
            parent_col = [-1] * n  # which row reached each column
            visited_rows[i] = True
            queue = [i]
            found = False

            while queue and not found:
                ri = queue.pop(0)
                for j in range(n):
                    if C[ri][j] != 0 or visited_cols[j]:
                        continue
                    visited_cols[j] = True
                    parent_col[j] = ri
                    if match_col[j] == -1:
                        # Augmenting path — flip matches along the path
                        col = j
                        while col != -1:
                            row = parent_col[col]
                            prev_col = match_row[row]
                            match_row[row] = col
                            match_col[col] = row
                            col = prev_col
                        found = True
                        break
                    else:
                        next_row = match_col[j]
                        if not visited_rows[next_row]:
                            visited_rows[next_row] = True
                            queue.append(next_row)

        if all(r != -1 for r in match_row):
            return sum(cost_matrix[i][match_row[i]] for i in range(n))

        # Step 4 – find minimum line cover, then adjust uncovered values.
        # Marked rows = unmatched rows, then propagate via zeros and matches.
        marked_rows = [match_row[i] == -1 for i in range(n)]
        marked_cols = [False] * n

        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not marked_rows[i]:
                    continue
                for j in range(n):
                    if C[i][j] == 0 and not marked_cols[j]:
                        marked_cols[j] = True
                        changed = True
            for j in range(n):
                if marked_cols[j] and match_col[j] != -1 and not marked_rows[match_col[j]]:
                    marked_rows[match_col[j]] = True
                    changed = True

        # Cover unmarked rows and marked columns
        covered_rows = [not marked_rows[i] for i in range(n)]
        covered_cols = marked_cols

        min_val = min(
            C[i][j]
            for i in range(n) for j in range(n)
            if not covered_rows[i] and not covered_cols[j]
        )
        for i in range(n):
            for j in range(n):
                if not covered_rows[i] and not covered_cols[j]:
                    C[i][j] -= min_val
                elif covered_rows[i] and covered_cols[j]:
                    C[i][j] += min_val


def boxes_to_goals_hungarian(state: State, level: Level) -> int:
    """Optimal 1-to-1 box-to-goal assignment cost (Hungarian algorithm)."""
    return _boxes_to_goals_hungarian_cached(state.boxes, level.goals)


@lru_cache(maxsize=None)
def _boxes_to_goals_hungarian_cached(
    boxes: FrozenSet[tuple[int, int]],
    goals: FrozenSet[tuple[int, int]],
) -> int:
    """Cached Hungarian assignment cost for a fixed box layout."""
    boxes = list(boxes)
    goals = list(goals)
    cost_matrix = [
        [manhattan(box, goal) for goal in goals]
        for box in boxes
    ]
    return _hungarian(cost_matrix)

def boxes_to_goals_hungarian_and_player(state: State, level: Level) -> int:
    """Hungarian assignment cost + player distance to nearest box."""
    return boxes_to_goals_hungarian_and_player_unsolved(state, level)


def boxes_to_goals_hungarian_and_player_all(state: State, level: Level) -> int:
    """Hungarian cost plus distance to any box, including solved boxes."""
    return (
        _boxes_to_goals_hungarian_cached(state.boxes, level.goals)
        + _nearest_box_distance(state, level, include_solved=True)
    )


def boxes_to_goals_hungarian_and_player_unsolved(
    state: State,
    level: Level,
) -> int:
    """Hungarian cost plus distance to the nearest unsolved box."""
    return (
        _boxes_to_goals_hungarian_cached(state.boxes, level.goals)
        + _nearest_unsolved_box_distance(state, level)
    )
