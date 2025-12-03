# tetris_env_vs_heuristic.py

import numpy as np
import random
from typing import Tuple

from tetris_env_rl import TetrisBoardRL, SHAPES, TETROMINOES


def heuristic_eval_board(board: np.ndarray) -> float:
    """
    휴리스틱 평가 H(s):
    - aggregate_height, holes, bumpiness 사용
    - 값이 클수록 좋은 상태
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
    - board: 0/1 numpy array
    - 주어진 (shape_idx, rotation, x)에 대해 최대한 아래로 떨어뜨린 후
      줄 제거까지 적용한 새로운 board를 리턴.
    - cleared_lines, illegal 여부도 반환.
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

    # 줄 제거
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


class HeuristicOpponent:
    """
    휴리스틱 테트리스 봇:
    - 가능한 (rot, x) 후보 중에서 휴리스틱 평가가 최대가 되는 수를 둔다.
    """

    def __init__(self, width=10):
        self.width = width

    def choose_action(self, board: TetrisBoardRL) -> Tuple[int, int]:
        """
        board.current_shape_idx 기준으로 최적 (rotation, x)를 선택.
        """
        shape_idx = board.current_shape_idx
        shape_name = SHAPES[shape_idx]
        num_rots = len(TETROMINOES[shape_name])

        best_score = -1e9
        best_rot = 0
        best_x = 0

        base_board = board.board  # np.ndarray (0/1)

        for rot in range(num_rots):
            for x in range(-2, self.width):
                sim_board, cleared, illegal = _simulate_drop(base_board, shape_idx, rot, x)
                if illegal:
                    continue

                h_score = heuristic_eval_board(sim_board)
                # 줄 제거에 대한 보너스도 약간 추가
                h_score += cleared * 10.0

                if h_score > best_score:
                    best_score = h_score
                    best_rot = rot
                    best_x = x

        return best_rot, best_x

    def place_piece(self, board: TetrisBoardRL) -> int:
        """
        실제 보드에 조각 배치.
        - board.current_shape_idx 기준
        - choose_action 호출 후 hard_drop_at 사용
        - 제거한 줄 수 반환
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
        if not over:
            # 새 조각
            board.current_shape_idx = -1
            board.spawn_new_piece()
        return cleared


class TetrisVsHeuristicEnv:
    """
    RL vs 휴리스틱 경쟁 환경.

    - RL 쪽: rl_board (TetrisBoardRL)
    - 휴리스틱 쪽: opp_board (TetrisBoardRL + HeuristicOpponent)

    - 상태: RL 보드(0/1 10x20) + RL 현재 조각 one-hot (207차원) → 싱글 환경과 동일
    - 행동: 0 ~ width*4-1 → (rot, x)

    - 리워드:
      r_env = (+ a1 * rl_cleared) - (a2 * opp_cleared) + 승패 보상
      r_shape = lambda * ( H(s') - H(s) )   # 휴리스틱 평가 기반 shaping

      최종 r = r_env + r_shape
    """

    def __init__(
        self,
        width=10,
        height=20,
        max_steps=500,
        seed: int = 0,
        a1: float = 1.0,
        a2: float = 0.5,
        win_bonus: float = 50.0,
        lose_penalty: float = 50.0,
        step_penalty: float = 0.01,
        shaping_lambda: float = 0.2,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.n_actions = width * 4

        self.rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.rl_board = TetrisBoardRL(width, height)
        self.opp_board = TetrisBoardRL(width, height)
        self.opp = HeuristicOpponent(width=width)

        self.steps = 0

        # 리워드 파라미터
        self.a1 = a1
        self.a2 = a2
        self.win_bonus = win_bonus
        self.lose_penalty = lose_penalty
        self.step_penalty = step_penalty
        self.shaping_lambda = shaping_lambda

    def _make_state(self) -> np.ndarray:
        """
        RL 상태:
        - rl_board.board (0/1 flatten 200)
        - rl_board.current_shape_idx one-hot (7)
        """
        board_flat = self.rl_board.board.reshape(-1).astype(np.float32)

        shape_onehot = np.zeros(len(SHAPES), dtype=np.float32)
        if self.rl_board.current_shape_idx >= 0:
            shape_onehot[self.rl_board.current_shape_idx] = 1.0

        state = np.concatenate([board_flat, shape_onehot], axis=0)  # 207
        return state

    def reset(self) -> np.ndarray:
        self.rl_board.reset()
        self.opp_board.reset()
        self.steps = 0

        # 양쪽 모두 첫 조각 spawn
        if self.rl_board.current_shape_idx < 0:
            self.rl_board.spawn_new_piece()
        if self.opp_board.current_shape_idx < 0:
            self.opp_board.spawn_new_piece()

        return self._make_state()

    def step(self, action: int):
        """
        returns: (next_state, reward, done, info)
        """
        self.steps += 1
        done = False
        info = {}

        # 이미 끝났으면
        if self.rl_board.game_over:
            return self._make_state(), 0.0, True, info

        # --- RL 쪽 한 수 두기 ---
        s_before = self.rl_board.board.copy()
        H_before = heuristic_eval_board(s_before)

        shape_idx = self.rl_board.current_shape_idx
        shape_name = SHAPES[shape_idx]
        num_rots = len(TETROMINOES[shape_name])

        rot_id = action // self.width
        x = action % self.width
        rotation = rot_id % num_rots

        cleared_rl, rl_illegal = self.rl_board.hard_drop_at(shape_idx, rotation, x)

        # RL board game over 체크
        if self.rl_board.game_over or rl_illegal:
            # 지는 경우
            reward_env = -self.lose_penalty
            done = True
            H_after = heuristic_eval_board(self.rl_board.board)
            reward_shape = self.shaping_lambda * (H_after - H_before)
            reward = reward_env + reward_shape
            next_state = self._make_state()
            # info에 몇 줄 지웠는지도 남겨주자
            info["cleared_rl"] = cleared_rl
            info["cleared_opp"] = 0
            info["result"] = "lose"
            return next_state, reward, done, info

        # RL 한 수 두고 줄 정리까지 끝난 상태에서 heuristic H(s') 계산
        H_after = heuristic_eval_board(self.rl_board.board)
        reward_shape = self.shaping_lambda * (H_after - H_before)

        # RL 쪽 줄 제거 보상
        reward_env = self.a1 * cleared_rl
        reward_env -= self.step_penalty  # 스텝 페널티

        # RL이 안 졌으면 상대(휴리스틱)도 수를 둔다
        if not self.opp_board.game_over:
            cleared_opp = self.opp.place_piece(self.opp_board)
        else:
            cleared_opp = 0

        reward_env -= self.a2 * cleared_opp

        # 승패 판정
        if self.opp_board.game_over and not self.rl_board.game_over:
            reward_env += self.win_bonus
            done = True
            result = "win"
        elif self.rl_board.game_over and not self.opp_board.game_over:
            reward_env -= self.lose_penalty
            done = True
            result = "lose"
        elif self.rl_board.game_over and self.opp_board.game_over:
            # 동시에 터진 경우: 무승부 정도로 처리
            done = True
            result = "draw"
        else:
            result = "ongoing"

        if self.steps >= self.max_steps and not done:
            done = True
            result = "timeout"

        reward = reward_env + reward_shape
        next_state = self._make_state()

        info["cleared_rl"] = cleared_rl
        info["cleared_opp"] = cleared_opp
        info["result"] = result

        return next_state, reward, done, info

    def sample_action(self) -> int:
        return self.rng.randint(0, self.n_actions - 1)
