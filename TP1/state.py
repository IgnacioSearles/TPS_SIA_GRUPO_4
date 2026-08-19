from typing import List, FrozenSet, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class Action(Enum):
    UP = ("U", 0, -1)
    DOWN = ("D", 0, 1)
    LEFT = ("L", -1, 0)
    RIGHT = ("R", 1, 0)

    def __init__(self, char: str, dx: int, dy: int):
        self.char = char
        self.dx = dx
        self.dy = dy

@dataclass
class Level:
    walls: FrozenSet[Tuple[int, int]]
    goals: FrozenSet[Tuple[int, int]]

class State:
    def __init__(self, player_pos: Tuple[int, int], boxes: FrozenSet[Tuple[int, int]], 
                 parent: Optional[State] = None, action: Optional[Action] = None, cost: int = 0):
        self.player_pos = player_pos
        self.boxes = boxes
        self.parent = parent
        self.action = action
        self.cost = cost

    def __hash__(self) -> int:
        return hash((self.player_pos, self.boxes))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return self.player_pos == other.player_pos and self.boxes == other.boxes

    def get_successors(self, level: Level) -> List['State']:
        """Generates valid child states (handling movement and basic deadlocks)."""
        succesors = []
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
                    blocked_vertical = (new_bx, new_by - 1) in level.walls or (new_bx, new_by + 1) in level.walls
                    blocked_horizontal = (new_bx - 1, new_by) in level.walls or (new_bx + 1, new_by) in level.walls

                    if blocked_vertical and blocked_horizontal:
                        continue

                new_boxes = frozenset((self.boxes - {new_player_pos}) | {new_box_pos})

            new_state = State(player_pos=new_player_pos, boxes=new_boxes, parent=self, action=action, cost=self.cost + 1)
            succesors.append(new_state)

        return succesors


    def is_goal(self, level: Level) -> bool:
        """Returns True if all boxes are on goals."""
        return self.boxes == level.goals