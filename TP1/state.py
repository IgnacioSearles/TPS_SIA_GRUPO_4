from typing import List, FrozenSet, Tuple
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
    def __init__(self):
        pass

    def __hash__(self) -> int:
        pass

    def __eq__(self, other: object) -> bool:
        pass

    def get_successors(self) -> List['State']:
        """Generates valid child states (handling movement and basic deadlocks)."""
        pass

    def is_goal(self) -> bool:
        """Returns True if all boxes are on goals."""
        pass