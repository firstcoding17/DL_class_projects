# tetris_env_rl.py
import numpy as np
import random
from typing import List, Tuple

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
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}

# ---------------------------
# 보드 & 환경
# ---------------------------


class TetrisBoardRL:
    """
    RL 학습 전용 단일 보드.
    - 보드는 0/1만 저장 (블록 유무만)
    - 현재 조각 종류는 별도 변수로 관리
    """

    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=np.int8)

        self.current_shape_idx: int = -1
        self.current_rotation: int = 0
        self.current_x: int = 0
        self.current_y: int = 0
        self.game_over: bool = False

    # --- 조각 관련 ---

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

    def spawn_new_piece(self):
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

    def hard_drop_at(self, shape_idx: int, rotation: int, x: int) -> Tuple[int, bool]:
        """
        주어진 (shape_idx, rotation, x)에 대해
        - 위에서부터 가능한 만큼 아래로 떨어뜨려 고정
        - 제거한 줄 수, game_over 여부 반환
        - 만약 애초에 올릴 수 없는 위치면 (0, True) 같은 식으로 강한 패널티 줄 수 있음
        """
        if not self._is_valid_position(shape_idx, rotation, x, 0):
            # 완전 말도 안 되는 위치 → illegal move로 간주
            self.game_over = True
            return 0, True

        y = 0
        while self._is_valid_position(shape_idx, rotation, x, y + 1):
            y += 1

        # 고정
        for bx, by in self._get_shape_cells(shape_idx, rotation, x, y):
            self.board[by, bx] = 1

        # 줄 제거
        cleared = self._clear_full_lines()
        # 고정 후 맨 위까지 차면 game_over
        if not self._is_valid_position(shape_idx, rotation, x, 0):
            self.game_over = True

        return cleared, self.game_over

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

    # --- 상태 표현 ---

    def get_state_vector(self) -> np.ndarray:
        """
        상태 벡터:
        - board: 10x20 → 200차원 (0/1 normalize)
        - current_shape one-hot: 7차원
        총 207차원
        """
        board_flat = self.board.reshape(-1).astype(np.float32)
        shape_onehot = np.zeros(len(SHAPES), dtype=np.float32)
        if self.current_shape_idx >= 0:
            shape_onehot[self.current_shape_idx] = 1.0
        state = np.concatenate([board_flat, shape_onehot], axis=0)
        return state

    def reset(self):
        self.board[:] = 0
        self.game_over = False
        self.current_shape_idx = -1
        self.current_rotation = 0
        self.current_x = 0
        self.current_y = 0
        self.spawn_new_piece()


class TetrisEnv:
    """
    간단 RL 환경:
    - action_space: 0 ~ (width * 4 - 1)
      → rot_id = a // width, x = a % width
      → 실제 rotation은 rot_id % num_rotations
    - step(action):
      - 현재 조각을 해당 (rotation, x)에 하드드롭
      - reward 계산 후 새 조각 스폰
    """

    def __init__(self, width=10, height=20, max_steps=500, seed: int = 0):
        self.board = TetrisBoardRL(width, height)
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.n_actions = width * 4
        self.rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.steps = 0

    def reset(self) -> np.ndarray:
        self.board.reset()
        self.steps = 0
        return self.board.get_state_vector()

    def step(self, action: int):
        """
        returns: (next_state, reward, done, info)
        """
        self.steps += 1
        done = False
        info = {}

        # 현재 조각 확인
        if self.board.game_over:
            done = True
            return self.board.get_state_vector(), 0.0, done, info

        shape_idx = self.board.current_shape_idx
        shape_name = SHAPES[shape_idx]
        num_rots = len(TETROMINOES[shape_name])

        # action → (rot, x)
        rot_id = action // self.width
        x = action % self.width
        rotation = rot_id % num_rots

        # 해당 위치에 하드드롭
        cleared, illegal_or_over = self.board.hard_drop_at(shape_idx, rotation, x)

        # 기본 보상: 줄 제거 보상 + 스텝 페널티
        reward = 0.0
        if cleared > 0:
            reward += (cleared ** 2) * 5.0  # 1줄:5, 2줄:20, 3줄:45, 4줄:80 ...
        reward -= 0.1  # 매 스텝마다 약간의 페널티

        if illegal_or_over:
            reward -= 10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        # 새 조각 스폰 (게임 안 끝났을 때만)
        if not done:
            self.board.current_shape_idx = -1
            self.board.spawn_new_piece()
            if self.board.game_over:
                reward -= 10.0
                done = True

        next_state = self.board.get_state_vector()
        return next_state, reward, done, info

    def sample_action(self) -> int:
        return self.rng.randint(0, self.n_actions - 1)
