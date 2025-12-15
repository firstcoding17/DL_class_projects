# tetris_env_vs_heuristic.py
import numpy as np
import random
from typing import Tuple

from tetris_env_rl import TetrisBoardRL, SHAPES, TETROMINOES


# ---------------------------
# 휴리스틱 평가 함수 (H(s))
# ---------------------------

def heuristic_eval_board(board: np.ndarray) -> float:
    """
    board: (h,w) 0/1
    - 높이, 홀수, bumpiness 기반 간단 휴리스틱
    """
    height, width = board.shape
    heights = [0] * width
    holes = 0

    for x in range(width):
        block_seen = False
        for y in range(height):
            if board[y, x] != 0:
                if not block_seen:
                    block_seen = True
                    heights[x] = height - y
            else:
                if block_seen:
                    holes += 1

    aggregate_height = sum(heights)
    bumpiness = 0
    for i in range(width - 1):
        bumpiness += abs(heights[i] - heights[i + 1])

    score = 0.0
    score -= aggregate_height * 0.5
    score -= holes * 5.0
    score -= bumpiness * 0.3
    return score


def _get_shape_cells(shape_idx: int, rotation: int, ox: int, oy: int):
    shape = SHAPES[shape_idx]
    cells = []
    for dx, dy in TETROMINOES[shape][rotation]:
        x = ox + dx
        y = oy + dy
        cells.append((x, y))
    return cells


def _is_valid(board: np.ndarray, shape_idx: int, rotation: int, ox: int, oy: int) -> bool:
    h, w = board.shape
    for x, y in _get_shape_cells(shape_idx, rotation, ox, oy):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        if board[y, x] != 0:
            return False
    return True


def _simulate_drop(board: np.ndarray, shape_idx: int, rotation: int, x: int) -> Tuple[np.ndarray, int, bool]:
    """
    휴리스틱용 시뮬레이션:
    - board: 0/1
    - (shape_idx, rotation, x)에 대해 하드드롭 → 줄 제거까지 적용
    - (새 보드, cleared_lines, illegal) 반환
    """
    h, w = board.shape
    new_board = board.copy()

    if not _is_valid(new_board, shape_idx, rotation, x, 0):
        return new_board, 0, True

    y = 0
    while _is_valid(new_board, shape_idx, rotation, x, y + 1):
        y += 1

    for bx, by in _get_shape_cells(shape_idx, rotation, x, y):
        new_board[by, bx] = 1

    rows = []
    cleared = 0
    for yy in range(h):
        if np.all(new_board[yy, :] != 0):
            cleared += 1
        else:
            rows.append(new_board[yy, :].copy())
    while len(rows) < h:
        rows.insert(0, np.zeros(w, dtype=np.int8))
    new_board = np.stack(rows, axis=0)

    return new_board, cleared, False


def calc_attack(cleared: int) -> int:
    """한 번에 k줄 제거 시 (k>=3) → (k-2)줄 공격."""
    return max(0, cleared - 2)


# ---------------------------
# 휴리스틱 상대 봇
# ---------------------------

class HeuristicOpponent:
    def __init__(self, width=10):
        self.width = width

    def choose_action(self, board: TetrisBoardRL):
        """
        board.current_shape_idx 기준으로 (rotation, x) 선택.
        """
        shape_idx = board.current_shape_idx
        shape_name = SHAPES[shape_idx]
        num_rots = len(TETROMINOES[shape_name])

        base_board = board.board
        best_score = -1e9
        best_rot = 0
        best_x = 0

        for rot in range(num_rots):
            for x in range(-2, self.width):
                sim_board, cleared, illegal = _simulate_drop(base_board, shape_idx, rot, x)
                if illegal:
                    continue
                h_score = heuristic_eval_board(sim_board)
                h_score += cleared * 10.0
                if h_score > best_score:
                    best_score = h_score
                    best_rot = rot
                    best_x = x

        return best_rot, best_x

    def place_piece(self, board: TetrisBoardRL) -> int:
        """
        휴리스틱이 실제로 한 수 둠. 제거한 줄 수 리턴.
        """
        if board.game_over:
            return 0

        if board.current_shape_idx < 0:
            board.spawn_new_piece()
            if board.game_over:
                return 0

        shape_idx = board.current_shape_idx
        rot, x = self.choose_action(board)
        cleared, over = board.hard_drop_at(shape_idx, rot, x)

        # 다음 턴을 위해 새 조각
        if not over:
            board.current_shape_idx = -1
            board.spawn_new_piece()

        return cleared


# ---------------------------
# RL vs 휴리스틱 환경
# ---------------------------

class TetrisVsHeuristicEnv:
    """
    - RL 보드 vs 휴리스틱 보드
    - state: RL 보드 상태 214차원 (board + current + hold)
    - action: 0~(width*4-1) 배치, width*4 = hold
    - reward:
      * 줄 제거(0.5 * cleared_rl)
      * 공격(2.0 * attack_rl)
      * 상대 공격(-1.0 * attack_opp)
      * 승리 +50, 패배 -50
      * 스텝 페널티 -0.01
      * + λ * ( H(s') - H(s) ) (작은 shaping)
    """

    def __init__(
        self,
        width=10,
        height=20,
        max_steps=500,
        seed: int = 0,
        win_bonus: float = 50.0,
        lose_penalty: float = 50.0,
        step_penalty: float = 0.01,
        shaping_lambda: float = 0.05,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.n_actions = width * 4 + 1  # +1: hold

        random.seed(seed)
        np.random.seed(seed)

        self.rl_board = TetrisBoardRL(width, height)
        self.opp_board = TetrisBoardRL(width, height)
        self.opp = HeuristicOpponent(width=width)

        self.steps = 0

        self.win_bonus = win_bonus
        self.lose_penalty = lose_penalty
        self.step_penalty = step_penalty
        self.shaping_lambda = shaping_lambda

    def reset(self) -> np.ndarray:
        self.rl_board.reset()
        self.opp_board.reset()
        self.steps = 0

        self.rl_board.spawn_new_piece()
        self.opp_board.spawn_new_piece()

        return self.rl_board.get_state_vector()

    def step(self, action: int):
        self.steps += 1
        done = False
        info = {}

        if self.rl_board.game_over:
            return self.rl_board.get_state_vector(), -self.lose_penalty, True, {"result": "already_over"}

        s_before = self.rl_board.board.copy()
        H_before = heuristic_eval_board(s_before)

        cleared_rl = 0
        attack_rl = 0
        cleared_opp = 0
        attack_opp = 0
        reward_env = 0.0
        result = "ongoing"

        # ------------------
        # RL 행동
        # ------------------
        if action == self.n_actions - 1:
            # hold
            self.rl_board.hold_current_piece()
            reward_env -= 0.05  # 홀드 비용
            if self.rl_board.game_over:
                reward_env -= self.lose_penalty
                done = True
                result = "lose"
        else:
            # 배치
            if self.rl_board.current_shape_idx < 0:
                self.rl_board.spawn_new_piece()
                if self.rl_board.game_over:
                    return self.rl_board.get_state_vector(), -self.lose_penalty, True, {"result": "lose"}

            shape_idx = self.rl_board.current_shape_idx
            shape_name = SHAPES[shape_idx]
            num_rots = len(TETROMINOES[shape_name])

            rot_id = action // self.width
            x = action % self.width
            rotation = rot_id % num_rots

            cleared_rl, illegal = self.rl_board.hard_drop_at(shape_idx, rotation, x)
            attack_rl = calc_attack(cleared_rl)

            reward_env += 0.5 * cleared_rl
            reward_env += 2.0 * attack_rl
            reward_env -= self.step_penalty

            if illegal:
                reward_env -= self.lose_penalty
                done = True
                result = "lose"
            else:
                # 새 조각 준비
                self.rl_board.current_shape_idx = -1
                self.rl_board.spawn_new_piece()
                if self.rl_board.game_over:
                    reward_env -= self.lose_penalty
                    done = True
                    result = "lose"

        # ------------------
        # 상대(휴리스틱) 행동
        # ------------------
        if not done and not self.opp_board.game_over:
            cleared_opp = self.opp.place_piece(self.opp_board)
            attack_opp = calc_attack(cleared_opp)
            reward_env -= 1.0 * attack_opp

        # 승패 판정
        if not done:
            if self.opp_board.game_over and not self.rl_board.game_over:
                reward_env += self.win_bonus
                done = True
                result = "win"
            elif self.rl_board.game_over and not self.opp_board.game_over:
                reward_env -= self.lose_penalty
                done = True
                result = "lose"
            elif self.rl_board.game_over and self.opp_board.game_over:
                done = True
                result = "draw"
            else:
                result = "ongoing"

        if self.steps >= self.max_steps and not done:
            done = True
            result = "timeout"

        H_after = heuristic_eval_board(self.rl_board.board)
        reward_shape = self.shaping_lambda * (H_after - H_before)
        reward = reward_env + reward_shape

        next_state = self.rl_board.get_state_vector()
        info["cleared_rl"] = cleared_rl
        info["attack_rl"] = attack_rl
        info["cleared_opp"] = cleared_opp
        info["attack_opp"] = attack_opp
        info["result"] = result

        return next_state, reward, done, info

    def sample_action(self) -> int:
        return random.randint(0, self.n_actions - 1)
