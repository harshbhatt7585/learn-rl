import random
from typing import Iterable, Optional, Set, Tuple


class GridWorld:
    """A slightly more complex gridworld compatible with the current PPO.

    Features:
    - Walls (impassable cells)
    - Traps (terminal with negative reward)
    - Stochastic slip: with probability `slip_prob`, action is replaced by a random action
    - Random starts (configurable)
    - Episode step limit

    Observation: single integer `current_place` in [0, size-1]
    Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        walls: Optional[Iterable[int]] = None,
        traps: Optional[Iterable[int]] = None,
        goal: Optional[int] = None,
        step_limit: int = 100,
        slip_prob: float = 0.1,
        random_start: bool = True,
        invalid_move_reward: float = -2.0,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        trap_reward: float = -10.0,
        start: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.size = width * height

        # Layout
        self.walls: Set[int] = set(walls or [])
        self.traps: Set[int] = set(traps or [])
        self.goal = self.size - 1 if goal is None else goal
        self.start = 0 if start is None else start
        self.random_start = random_start

        # Dynamics and episode config
        self.step_limit = step_limit
        self.slip_prob = slip_prob

        # Rewards
        self.invalid_move_reward = invalid_move_reward
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward

        # Runtime state
        self.actions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.current_place = 0
        self.step_count = 0

        # Provide a reasonable default layout if none specified
        if walls is None and traps is None and goal is None:
            # A small maze-ish layout for 5x5
            default_walls = {7, 12, 13, 17}
            default_traps = {6, 18}
            self.walls |= default_walls
            self.traps |= default_traps

    # --- Helpers ---
    def rc_to_idx(self, row: int, col: int) -> int:
        return row * self.width + col

    def idx_to_rc(self, idx: int) -> Tuple[int, int]:
        return idx // self.width, idx % self.width

    def _is_in_bounds(self, idx: int) -> bool:
        return 0 <= idx < self.size

    def _is_wall(self, idx: int) -> bool:
        return idx in self.walls

    # --- API ---
    def reset(self) -> int:
        self.step_count = 0
        if self.random_start:
            candidates = [i for i in range(self.size) if i not in self.walls and i not in self.traps and i != self.goal]
            self.current_place = random.choice(candidates) if candidates else 0
        else:
            self.current_place = self.start
        return self.current_place

    def step(self, action: int):
        self.step_count += 1

        # Slip: with prob slip_prob replace action with random action
        if random.random() < self.slip_prob:
            action = random.randint(0, 3)

        row, col = self.idx_to_rc(self.current_place)
        if action == 0:  # UP
            new_idx = self.rc_to_idx(row - 1, col)
        elif action == 1:  # DOWN
            new_idx = self.rc_to_idx(row + 1, col)
        elif action == 2:  # LEFT
            new_idx = self.rc_to_idx(row, col - 1)
        elif action == 3:  # RIGHT
            new_idx = self.rc_to_idx(row, col + 1)
        else:
            new_idx = self.current_place

        valid = self._is_in_bounds(new_idx) and not self._is_wall(new_idx)

        # Determine transition and reward
        if not valid:
            reward = self.invalid_move_reward
            next_state = self.current_place
        else:
            self.current_place = new_idx
            next_state = self.current_place
            reward = self.step_reward

        done = False
        if next_state in self.traps:
            reward += self.trap_reward
            done = True
        elif next_state == self.goal:
            reward += self.goal_reward
            done = True
        elif self.step_count >= self.step_limit:
            done = True

        return next_state, reward, done, self.step_count


if __name__ == "__main__":
    env = GridWorld()
    s = env.reset()
    total = 0
    for _ in range(20):
        s, r, d, t = env.step(random.randint(0, 3))
        total += r
        if d:
            break
    print("state:", s, "reward:", total, "done:", d)


