from collections import deque
from functools import lru_cache
from typing import List, FrozenSet, Tuple
from enum import Enum
from dataclasses import dataclass

PRUNING_MODES = {"dead_squares", "local"}

class Action(Enum):
    UP = ("U", 0, -1)
    DOWN = ("D", 0, 1)
    LEFT = ("L", -1, 0)
    RIGHT = ("R", 1, 0)

    def __init__(self, char: str, dx: int, dy: int):
        self.char = char
        self.dx = dx
        self.dy = dy

@dataclass(frozen=True)
class Level:
    walls: FrozenSet[Tuple[int, int]]
    goals: FrozenSet[Tuple[int, int]]


@lru_cache(maxsize=None)
def _dead_squares(
    walls: FrozenSet[Tuple[int, int]],
    goals: FrozenSet[Tuple[int, int]],
) -> FrozenSet[Tuple[int, int]]:
    """Return non-goal cells from which no goal can be reached by pushes."""
    known_cells = walls | goals
    if not known_cells:
        return frozenset()

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

    reachable = set(goals)
    frontier = deque(goals)
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))

    while frontier:
        current_x, current_y = frontier.popleft()

        for dx, dy in directions:
            predecessor = (current_x - dx, current_y - dy)
            support = (current_x - 2 * dx, current_y - 2 * dy)

            if predecessor not in floor or support not in floor:
                continue
            if predecessor in reachable:
                continue

            reachable.add(predecessor)
            frontier.append(predecessor)

    return frozenset(floor - reachable - goals)


@dataclass(frozen=True)
class State:
    player_pos: Tuple[int, int]
    boxes: FrozenSet[Tuple[int, int]]

    def get_successors(
        self,
        level: Level,
        pruning_mode: str = "dead_squares",
    ) -> List[Tuple['State', Action, int]]:
        """Generates valid child states with the action and step cost."""
        if pruning_mode not in PRUNING_MODES:
            valid_modes = ", ".join(sorted(PRUNING_MODES))
            raise ValueError(
                f"Unknown pruning mode '{pruning_mode}'. Valid options: {valid_modes}"
            )

        successors = []
        px, py = self.player_pos

        for action in Action:
            new_px, new_py = px + action.dx, py + action.dy
            new_player_pos = (new_px, new_py)

            if new_player_pos in level.walls:
                continue

            new_boxes = self.boxes

            if new_player_pos in self.boxes:
                new_bx, new_by = new_px + action.dx, new_py + action.dy  
                new_box_pos = (new_bx, new_by)

                if new_box_pos in self.boxes or new_box_pos in level.walls:
                    continue

                # Deadlock detection
                if new_box_pos not in level.goals:
                    if (
                        pruning_mode == "dead_squares"
                        and new_box_pos in _dead_squares(level.walls, level.goals)
                    ):
                        continue

                    if pruning_mode == "local":
                        blocked_vertical = (
                            (new_bx, new_by - 1) in level.walls
                            or (new_bx, new_by + 1) in level.walls
                        )
                        blocked_horizontal = (
                            (new_bx - 1, new_by) in level.walls
                            or (new_bx + 1, new_by) in level.walls
                        )

                        if blocked_vertical and blocked_horizontal:
                            continue

                new_boxes = frozenset((self.boxes - {new_player_pos}) | {new_box_pos})

            new_state = State(player_pos=new_player_pos, boxes=new_boxes)
            successors.append((new_state, action, 1))

        return successors


    def is_goal(self, level: Level) -> bool:
        """Returns True if all boxes are on goals."""
        return self.boxes == level.goals
