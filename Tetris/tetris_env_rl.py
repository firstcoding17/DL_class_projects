# tetris_env_rl.py
import numpy as np
import random
from typing import Tuple

# ---------------------------
# 테트리스 조각 정의
# ---------------------------

TETROMINOES = {
    "I": [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
    ],
    "O": [
        [(1, 0), (2, 0), (1, 1), (2, 1)],
    ],
    "T": [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "S": [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
    ],
    "Z": [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
    ],
    "J": [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    "L": [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
}
SHAPES = list(TETROMINOES.keys())


# ---------------------------
# 보드 (RL용)
# ---------------------------

class TetrisBoardRL:
    """
    - board: (height, width) 0/1
    - current_shape_idx: 0~6, -1이면 없음
    - hold_shape_idx: 0~6, -1이면 홀드 비어있음
    - can_hold: 이번 조각에서 홀드 가능 여부
    """

    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=np.int8)

        self.current_shape_idx: int = -1
        self.current_rotation: int = 0
        self.current_x: int = 0
        self.current_y: int = 0

        self.hold_shape_idx: int = -1
        self.can_hold: bool = True
        self.game_over: bool = False

    # --- 내부 유틸 ---

    def _get_shape_cells(self, shape_idx: int, rotation: int, ox: int, oy: int):
        shape = SHAPES[shape_idx]
        cells = []
        for dx, dy in TETROMINOES[shape][rotation]:
            x = ox + dx
            y = oy + dy
            cells.append((x, y))
        return cells

    def _is_valid_position(self, shape_idx: int, rotation: int, ox: int, oy: int) -> bool:
        for x, y in self._get_shape_cells(shape_idx, rotation, ox, oy):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return False
            if self.board[y, x] != 0:
                return False
        return True

    # --- 조각 관련 ---

    def spawn_new_piece(self):
        """새 조각 스폰. 못 놓으면 game_over."""
        shape_idx = random.randint(0, len(SHAPES) - 1)
        rotation = 0
        ox = self.width // 2 - 2
        oy = 0

        if not self._is_valid_position(shape_idx, rotation, ox, oy):
            self.game_over = True
            return

        self.current_shape_idx = shape_idx
        self.current_rotation = rotation
        self.current_x = ox
        self.current_y = oy
        self.can_hold = True  # 새 조각 나오면 다시 홀드 가능

    def hard_drop_at(self, shape_idx: int, rotation: int, x: int) -> Tuple[int, bool]:
        """
        (shape_idx, rotation, x)에 대해 최대한 아래로 떨어뜨려 고정.
        - 줄 제거 수, game_over 여부 반환
        - 애초에 불가능한 위치면 (0, True) 로 간주
        """
        if not self._is_valid_position(shape_idx, rotation, x, 0):
            self.game_over = True
            return 0, True

        y = 0
        while self._is_valid_position(shape_idx, rotation, x, y + 1):
            y += 1

        for bx, by in self._get_shape_cells(shape_idx, rotation, x, y):
            self.board[by, bx] = 1

        cleared = self._clear_full_lines()

        # 맨 위까지 차면 패배
        if not self._is_valid_position(shape_idx, rotation, x, 0):
            self.game_over = True
            return cleared, True

        return cleared, False

    def _clear_full_lines(self) -> int:
        new_rows = []
        cleared = 0
        for y in range(self.height):
            if np.all(self.board[y, :] != 0):
                cleared += 1
            else:
                new_rows.append(self.board[y, :].copy())
        while len(new_rows) < self.height:
            new_rows.insert(0, np.zeros(self.width, dtype=np.int8))
        self.board = np.stack(new_rows, axis=0)
        return cleared

    # --- 홀드 기능 ---

    def hold_current_piece(self):
        """
        현재 조각을 홀드.
        - 할 수 없으면 아무 일도 안 일어남.
        - 홀드가 비어 있으면: 현재 조각을 홀드로 보내고 새 조각 스폰
        - 이미 홀드가 있으면: 현재 조각과 교체
        """
        if not self.can_hold or self.current_shape_idx < 0 or self.game_over:
            return

        shape_to_hold = self.current_shape_idx

        if self.hold_shape_idx < 0:
            # 홀드 비어 있음 → 현재 조각을 홀드로 보내고 새 조각 스폰
            self.hold_shape_idx = shape_to_hold
            self.current_shape_idx = -1
            self.current_rotation = 0
            self.current_x = 0
            self.current_y = 0
            self.spawn_new_piece()
        else:
            # 홀드에 조각 있음 → 현재 조각과 교체
            new_shape = self.hold_shape_idx
            ox = self.width // 2 - 2
            oy = 0
            if self._is_valid_position(new_shape, 0, ox, oy):
                self.hold_shape_idx = shape_to_hold
                self.current_shape_idx = new_shape
                self.current_rotation = 0
                self.current_x = ox
                self.current_y = oy
            else:
                self.game_over = True

        self.can_hold = False

    # --- 상태 벡터 ---

    def get_state_vector(self) -> np.ndarray:
        """
        state: [board_flat(200), current_shape_onehot(7), hold_shape_onehot(7)] = 214
        """
        board_flat = self.board.reshape(-1).astype(np.float32)

        cur_onehot = np.zeros(len(SHAPES), dtype=np.float32)
        if self.current_shape_idx >= 0:
            cur_onehot[self.current_shape_idx] = 1.0

        hold_onehot = np.zeros(len(SHAPES), dtype=np.float32)
        if self.hold_shape_idx >= 0:
            hold_onehot[self.hold_shape_idx] = 1.0

        state = np.concatenate([board_flat, cur_onehot, hold_onehot], axis=0)
        return state

    def reset(self):
        self.board[:] = 0
        self.current_shape_idx = -1
        self.current_rotation = 0
        self.current_x = 0
        self.current_y = 0
        self.hold_shape_idx = -1
        self.can_hold = True
        self.game_over = False


# ---------------------------
# 싱글 RL 환경
# ---------------------------

class TetrisEnv:
    """
    - action space:
      0 ~ (width*4 - 1): (rotation, x) 배치
      width*4: 홀드 액션 (이번 턴은 홀드만 하고 조각 안 둠)
    """

    def __init__(self, width=10, height=20, max_steps=500, seed: int = 0):
        self.board = TetrisBoardRL(width, height)
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.n_actions = width * 4 + 1  # +1: hold 액션

        random.seed(seed)
        np.random.seed(seed)

        self.steps = 0

    def reset(self) -> np.ndarray:
        self.board.reset()
        self.steps = 0
        self.board.spawn_new_piece()
        return self.board.get_state_vector()

    def step(self, action: int):
        """
        returns: (next_state, reward, done, info)
        """
        self.steps += 1
        done = False
        info = {}

        if self.board.game_over:
            return self.board.get_state_vector(), 0.0, True, info

        reward = 0.0

        # Hold 액션
        if action == self.n_actions - 1:
            self.board.hold_current_piece()
            if self.board.game_over:
                reward -= 10.0
                done = True
            else:
                reward -= 0.05  # 홀드 비용
        else:
            # 배치 액션
            if self.board.current_shape_idx < 0:
                self.board.spawn_new_piece()
                if self.board.game_over:
                    return self.board.get_state_vector(), -10.0, True, info

            shape_idx = self.board.current_shape_idx
            shape_name = SHAPES[shape_idx]
            num_rots = len(TETROMINOES[shape_name])

            rot_id = action // self.width
            x = action % self.width
            rotation = rot_id % num_rots

            cleared, illegal = self.board.hard_drop_at(shape_idx, rotation, x)
            reward += cleared * 1.0
            reward -= 0.01

            if illegal:
                reward -= 10.0
                done = True
            else:
                # 다음 턴을 위해 새 조각
                self.board.current_shape_idx = -1
                self.board.spawn_new_piece()
                if self.board.game_over:
                    reward -= 10.0
                    done = True

        if self.steps >= self.max_steps and not done:
            done = True

        next_state = self.board.get_state_vector()
        return next_state, reward, done, info

    def sample_action(self) -> int:
        return random.randint(0, self.n_actions - 1)

