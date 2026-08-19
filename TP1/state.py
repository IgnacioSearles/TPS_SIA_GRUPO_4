from typing import List, FrozenSet, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class Action(Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"

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
        pass

    def __eq__(self, other: object) -> bool:
        pass

    def get_successors(self, level: Level) -> List['State']:
        """Generates valid child states (handling movement and basic deadlocks)."""
        pass

    def is_goal(self, level: Level) -> bool:
        """Returns True if all boxes are on goals."""
        pass