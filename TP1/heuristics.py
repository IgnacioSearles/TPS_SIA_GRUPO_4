from state import State, Level

def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ── Simple (non-exclusive) ────────────────────────────────────────────────────

def boxes_to_goals(state: State, level: Level) -> int:
    """Sum of each box's distance to its nearest goal (goals can be shared)."""
    return sum(
        min(manhattan(box, goal) for goal in level.goals)
        for box in state.boxes
    )

def boxes_to_goals_and_player(state: State, level: Level) -> int:
    """boxes_to_goals + player distance to nearest box."""
    if not state.boxes:
        return 0
    nearest_box = min(manhattan(state.player_pos, box) for box in state.boxes)
    return boxes_to_goals(state, level) + nearest_box


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
    boxes = list(state.boxes)
    goals = list(level.goals)
    cost_matrix = [
        [manhattan(box, goal) for goal in goals]
        for box in boxes
    ]
    return _hungarian(cost_matrix)

def boxes_to_goals_hungarian_and_player(state: State, level: Level) -> int:
    """Hungarian assignment cost + player distance to nearest box."""
    if not state.boxes:
        return 0
    nearest_box = min(manhattan(state.player_pos, box) for box in state.boxes)
    return boxes_to_goals_hungarian(state, level) + nearest_box
