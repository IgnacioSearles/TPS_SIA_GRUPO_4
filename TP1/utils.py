
from state import State, Level

def load_level(filepath: str) -> tuple[State, Level]:
    walls = set()
    goals = set()
    boxes = set()
    player_pos = None

    with open(filepath) as f:
        for y, line in enumerate(f):
            for x, ch in enumerate(line.rstrip('\n')):
                pos = (x, y)
                if ch == '#':
                    walls.add(pos)
                elif ch == '.':
                    goals.add(pos)
                elif ch == '$':
                    boxes.add(pos)
                elif ch == '@':
                    player_pos = pos
                elif ch == '+':   # player on goal
                    player_pos = pos
                    goals.add(pos)
                elif ch == '*':   # box on goal
                    boxes.add(pos)
                    goals.add(pos)

    if player_pos is None:
        raise ValueError(f"No player found in {filepath}")

    level = Level(walls=frozenset(walls), goals=frozenset(goals))
    initial_state = State(player_pos=player_pos, boxes=frozenset(boxes))
    return initial_state, level