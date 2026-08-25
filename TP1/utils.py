from state import State, Level
import time
import os

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


def render_board(level: Level, state: State) -> str:
    known_cells = level.walls | level.goals | state.boxes | {state.player_pos}
    if not known_cells:
        return ""
    
    min_x = min(x for x, _ in known_cells)
    max_x = max(x for x, _ in known_cells)
    min_y = min(y for _, y in known_cells)
    max_y = max(y for _, y in known_cells)
    
    lines = []
    for y in range(min_y, max_y + 1):
        line = []
        for x in range(min_x, max_x + 1):
            pos = (x, y)
            if pos == state.player_pos:
                if pos in level.goals:
                    line.append('+')
                else:
                    line.append('@')
            elif pos in state.boxes:
                if pos in level.goals:
                    line.append('*')
                else:
                    line.append('$')
            elif pos in level.goals:
                line.append('.')
            elif pos in level.walls:
                line.append('#')
            else:
                line.append(' ')
        lines.append("".join(line))
    return "\n".join(lines)

def animate_solution(initial_state: State, level: Level, path: list, delay: float = 0.2):
    current_state = initial_state
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Initial State")
    print(render_board(level, current_state))
    time.sleep(delay)
    
    for action in path:
        px, py = current_state.player_pos
        new_px, new_py = px + action.dx, py + action.dy
        new_player_pos = (new_px, new_py)
        
        new_boxes = current_state.boxes
        if new_player_pos in current_state.boxes:
            new_bx, new_by = new_px + action.dx, new_py + action.dy
            new_box_pos = (new_bx, new_by)
            new_boxes = frozenset((current_state.boxes - {new_player_pos}) | {new_box_pos})
            
        current_state = State(player_pos=new_player_pos, boxes=new_boxes)
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Action: {action.name}")
        print(render_board(level, current_state))
        time.sleep(delay)
