import sys
import random
import math
import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel
)
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QSize, QTimer

import os


# =========================
# 테트리스 조각 정의
# =========================

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

COLOR_MAP = {
    "I": QColor(0, 255, 255),
    "O": QColor(255, 255, 0),
    "T": QColor(128, 0, 128),
    "S": QColor(0, 255, 0),
    "Z": QColor(255, 0, 0),
    "J": QColor(0, 0, 255),
    "L": QColor(255, 165, 0),
}


# =========================
# 보드 및 게임 로직
# =========================

class TetrisBoard:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        # "" = 빈칸, 그 외 = 모양 문자("I","O"...)
        self.board: List[List[str]] = [
            ["" for _ in range(width)] for _ in range(height)
        ]

        # 현재 조각
        self.current_shape: str = ""
        self.current_rotation: int = 0
        self.current_x: int = 0
        self.current_y: int = 0

        # 홀드(keep)
        self.hold_shape: str = ""
        self.can_hold: bool = True

        self.game_over: bool = False

    # ---------- 유틸 ----------

    def clone_board_matrix(self) -> List[List[str]]:
        return copy.deepcopy(self.board)

    def get_shape_cells(self, shape: str, rotation: int, ox: int, oy: int) -> List[Tuple[int, int]]:
        cells = []
        for dx, dy in TETROMINOES[shape][rotation]:
            x = ox + dx
            y = oy + dy
            cells.append((x, y))
        return cells

    def is_valid_position(self, shape: str, rotation: int, ox: int, oy: int) -> bool:
        for x, y in self.get_shape_cells(shape, rotation, ox, oy):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return False
            if self.board[y][x] != "":
                return False
        return True

    # ---------- 조각 제어 ----------

    def spawn_new_piece(self):
        """새 조각 생성. 못 놓으면 game_over."""
        shape = random.choice(SHAPES)
        rotation = 0
        ox = self.width // 2 - 2
        oy = 0

        if not self.is_valid_position(shape, rotation, ox, oy):
            self.game_over = True
            return

        self.current_shape = shape
        self.current_rotation = rotation
        self.current_x = ox
        self.current_y = oy
        self.can_hold = True  # 새 조각이 나오면 다시 홀드 가능

    def move(self, dx: int, dy: int) -> bool:
        if not self.current_shape:
            return False
        nx = self.current_x + dx
        ny = self.current_y + dy
        if self.is_valid_position(self.current_shape, self.current_rotation, nx, ny):
            self.current_x = nx
            self.current_y = ny
            return True
        return False

    def rotate(self, direction: int = 1) -> bool:
        if not self.current_shape:
            return False
        nrot = (self.current_rotation + direction) % len(TETROMINOES[self.current_shape])
        if self.is_valid_position(self.current_shape, nrot, self.current_x, self.current_y):
            self.current_rotation = nrot
            return True
        return False

    def hard_drop(self) -> int:
        if not self.current_shape:
            return 0
        while self.move(0, 1):
            pass
        cleared = self.lock_piece()
        return cleared

    def lock_piece(self) -> int:
        if not self.current_shape:
            return 0
        for x, y in self.get_shape_cells(self.current_shape, self.current_rotation, self.current_x, self.current_y):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.board[y][x] = self.current_shape
            else:
                self.game_over = True

        self.current_shape = ""
        self.current_rotation = 0
        self.current_x = 0
        self.current_y = 0

        cleared = self.clear_full_lines()
        return cleared

    def clear_full_lines(self) -> int:
        new_rows = []
        cleared = 0
        for row in self.board:
            if all(cell != "" for cell in row):
                cleared += 1
            else:
                new_rows.append(row)

        while len(new_rows) < self.height:
            new_rows.insert(0, ["" for _ in range(self.width)])

        self.board = new_rows
        return cleared

    # ---------- 홀드 ----------

    def hold_current_piece(self):
        if not self.can_hold or not self.current_shape:
            return

        shape_to_hold = self.current_shape

        if self.hold_shape == "":
            self.hold_shape = shape_to_hold
            self.current_shape = ""
            self.current_rotation = 0
            self.current_x = 0
            self.current_y = 0
            self.spawn_new_piece()
        else:
            new_shape = self.hold_shape
            ox = self.width // 2 - 2
            oy = 0
            if self.is_valid_position(new_shape, 0, ox, oy):
                self.hold_shape = shape_to_hold
                self.current_shape = new_shape
                self.current_rotation = 0
                self.current_x = ox
                self.current_y = oy
            else:
                self.game_over = True

        self.can_hold = False

    # ---------- 공격(쓰레기 줄) ----------

    def add_garbage(self, n_lines: int, easy: bool = True):
        if n_lines <= 0:
            return

        hole_col = random.randint(self.width // 3, self.width - self.width // 3 - 1)

        for _ in range(n_lines):
            self.board.pop(0)

            row = [random.choice(SHAPES) for _ in range(self.width)]
            if easy:
                row[hole_col] = ""
            else:
                idx = random.randint(0, self.width - 1)
                row[idx] = ""

            self.board.append(row)

    # ---------- 평가 (휴리스틱용) ----------

    @staticmethod
    def compute_heights_and_holes(board: List[List[str]]) -> Tuple[List[int], int]:
        width = len(board[0])
        height = len(board)
        heights = [0] * width
        holes = 0

        for x in range(width):
            column_block_seen = False
            for y in range(height):
                if board[y][x] != "":
                    if not column_block_seen:
                        column_block_seen = True
                        heights[x] = height - y
                else:
                    if column_block_seen:
                        holes += 1
        return heights, holes

    @staticmethod
    def evaluate_board(board: List[List[str]], lines_cleared: int) -> float:
        heights, holes = TetrisBoard.compute_heights_and_holes(board)
        aggregate_height = sum(heights)

        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])

        score = 0.0
        score += lines_cleared * 10.0
        score -= aggregate_height * 0.5
        score -= holes * 5.0
        score -= bumpiness * 0.3

        return score


# -------------------------
# static helper for AI
# -------------------------

def _get_shape_cells_static(shape: str, rotation: int, ox: int, oy: int):
    cells = []
    for dx, dy in TETROMINOES[shape][rotation]:
        x = ox + dx
        y = oy + dy
        cells.append((x, y))
    return cells


def _is_valid_position_static(board_matrix: List[List[str]], shape: str, rotation: int, ox: int, oy: int) -> bool:
    h = len(board_matrix)
    w = len(board_matrix[0])
    for x, y in _get_shape_cells_static(shape, rotation, ox, oy):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        if board_matrix[y][x] != "":
            return False
    return True


TetrisBoard.get_shape_cells_static = staticmethod(_get_shape_cells_static)
TetrisBoard.is_valid_position_static = staticmethod(_is_valid_position_static)


# =========================
# 휴리스틱 AI
# =========================

class TetrisAI:
    def choose_and_place(self, board: TetrisBoard) -> int:
        if not board.current_shape:
            board.spawn_new_piece()
            if board.game_over:
                return 0

        shape = board.current_shape
        best_score = -math.inf
        best_x = board.current_x
        best_rot = board.current_rotation

        for rot_idx in range(len(TETROMINOES[shape])):
            for x in range(-2, board.width):
                y = 0
                sim_board = board.clone_board_matrix()
                if not TetrisBoard.is_valid_position_static(sim_board, shape, rot_idx, x, y):
                    continue

                while True:
                    if not TetrisBoard.is_valid_position_static(sim_board, shape, rot_idx, x, y + 1):
                        break
                    y += 1

                for bx, by in TetrisBoard.get_shape_cells_static(shape, rot_idx, x, y):
                    sim_board[by][bx] = shape

                sim_board, lines_cleared = TetrisAI.sim_clear_lines(sim_board)
                score = TetrisBoard.evaluate_board(sim_board, lines_cleared)

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_rot = rot_idx

        board.current_rotation = best_rot
        board.current_x = best_x
        board.current_y = 0
        cleared = board.hard_drop()
        return cleared

    @staticmethod
    def sim_clear_lines(board: List[List[str]]) -> Tuple[List[List[str]], int]:
        height = len(board)
        width = len(board[0])
        new_rows = []
        cleared = 0
        for row in board:
            if all(cell != "" for cell in row):
                cleared += 1
            else:
                new_rows.append(row)
        while len(new_rows) < height:
            new_rows.insert(0, ["" for _ in range(width)])
        return new_rows, cleared


# =========================
# RL Policy (train_tetris_rl.py와 동일 구조)
# =========================

class PolicyNet(nn.Module):
    def __init__(self, state_dim=207, n_actions=40, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class RLTetrisAI:
    """
    RL 정책으로 (rot, x) 선택해서 한 번에 조각을 두는 봇.
    """

    def __init__(self, board_width: int, model_path: str = "tetris_policy.pth"):
        self.width = board_width
        self.state_dim = 10 * 20 + len(SHAPES)  # 200 + 7 = 207
        self.n_actions = self.width * 4
        self.device = torch.device("cpu")

        self.policy = PolicyNet(self.state_dim, self.n_actions).to(self.device)

        if not os.path.exists(model_path):
            print(f"[RLTetrisAI] WARNING: 모델 파일 {model_path} 없음 → 휴리스틱 fallback")
            self.policy = None
        else:
            state_dict = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
            self.policy.eval()
            print(f"[RLTetrisAI] Loaded model from {model_path}")

        self.backup_ai = TetrisAI()

    def _make_state(self, board: TetrisBoard) -> np.ndarray:
        h = board.height
        w = board.width
        mat = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                if board.board[y][x] != "":
                    mat[y, x] = 1.0
        flat = mat.reshape(-1)  # 200

        shape_onehot = np.zeros(len(SHAPES), dtype=np.float32)
        if board.current_shape:
            idx = SHAPES.index(board.current_shape)
            shape_onehot[idx] = 1.0

        state = np.concatenate([flat, shape_onehot], axis=0)
        return state

    def choose_and_place(self, board: TetrisBoard) -> int:
        if self.policy is None:
            return self.backup_ai.choose_and_place(board)

        if not board.current_shape:
            board.spawn_new_piece()
            if board.game_over:
                return 0

        state = self._make_state(board)
        s_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(s_tensor)
            action = torch.argmax(logits, dim=1).item()

        rot_id = action // self.width
        x = action % self.width

        shape = board.current_shape
        num_rots = len(TETROMINOES[shape])
        rotation = rot_id % num_rots

        if not board.is_valid_position(shape, rotation, x, 0):
            return self.backup_ai.choose_and_place(board)

        board.current_rotation = rotation
        board.current_x = x
        board.current_y = 0
        cleared = board.hard_drop()
        return cleared


# =========================
# PyQt 보드 위젯
# =========================

class BoardWidget(QWidget):
    def __init__(self, board: TetrisBoard, parent=None):
        super().__init__(parent)
        self.board = board
        self.cell_size = 24
        self.setMinimumSize(
            QSize(self.board.width * self.cell_size, self.board.height * self.cell_size)
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        pen = QPen(QColor(40, 40, 40))
        painter.setPen(pen)

        for y in range(self.board.height):
            for x in range(self.board.width):
                rect_x = x * self.cell_size
                rect_y = y * self.cell_size
                painter.drawRect(rect_x, rect_y, self.cell_size, self.cell_size)

                cell = self.board.board[y][x]
                if cell != "":
                    color = COLOR_MAP.get(cell, QColor(200, 200, 200))
                    painter.fillRect(
                        rect_x + 1,
                        rect_y + 1,
                        self.cell_size - 1,
                        self.cell_size - 1,
                        color,
                    )

        if self.board.current_shape:
            painter.setPen(Qt.NoPen)
            color = COLOR_MAP.get(self.board.current_shape, QColor(200, 200, 200))
            for x, y in self.board.get_shape_cells(
                self.board.current_shape,
                self.board.current_rotation,
                self.board.current_x,
                self.board.current_y,
            ):
                if 0 <= x < self.board.width and 0 <= y < self.board.height:
                    rect_x = x * self.cell_size
                    rect_y = y * self.cell_size
                    painter.fillRect(
                        rect_x + 1,
                        rect_y + 1,
                        self.cell_size - 1,
                        self.cell_size - 1,
                        color,
                    )


# =========================
# 메인 윈도우
#   vs_mode:
#     - "human_vs_heuristic"
#     - "human_vs_rl"
#     - "rl_vs_rl"
# =========================

class MainWindow(QWidget):
    def __init__(self, vs_mode: str = "human_vs_heuristic"):
        super().__init__()

        self.vs_mode = vs_mode

        if vs_mode == "human_vs_heuristic":
            title_mode = "휴리스틱"
        elif vs_mode == "human_vs_rl":
            title_mode = "강화학습(RL)"
        else:
            title_mode = "RL vs RL"

        self.setWindowTitle(f"PyQt5 테트리스 - 모드: {title_mode}")

        self.player_board = TetrisBoard()
        self.ai_board = TetrisBoard()

        # AI 설정
        if vs_mode == "human_vs_heuristic":
            self.ai = TetrisAI()
            self.rl_left = None
            self.rl_right = None
        elif vs_mode == "human_vs_rl":
            self.ai = RLTetrisAI(self.ai_board.width, model_path="tetris_policy.pth")
            self.rl_left = None
            self.rl_right = None
        else:  # rl_vs_rl
            self.ai = None
            self.rl_left = RLTetrisAI(self.player_board.width, model_path="tetris_policy.pth")
            self.rl_right = RLTetrisAI(self.ai_board.width, model_path="tetris_policy.pth")

        self.player_widget = BoardWidget(self.player_board)
        self.ai_widget = BoardWidget(self.ai_board)

        if "human" in vs_mode:
            info_text = "조작: ←/→ 좌우, ↑ 회전, ↓ 한 칸, Space 하드드롭, C 홀드"
        else:
            info_text = "RL vs RL 자동 대전 (키 입력 없음)"

        self.info_label = QLabel(info_text)
        self.status_label = QLabel("")
        self.player_hold_label = QLabel("Player Hold: -")
        self.ai_hold_label = QLabel(f"AI Hold: - (모드: {title_mode})")

        boards_layout = QHBoxLayout()
        boards_layout.addWidget(self.player_widget)
        boards_layout.addWidget(self.ai_widget)

        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addLayout(boards_layout)
        layout.addWidget(self.player_hold_label)
        layout.addWidget(self.ai_hold_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        # 점수 & 속도
        self.player_score = 0
        self.ai_score = 0
        self.total_cleared_by_player = 0
        self.level = 1
        self.drop_interval = 800
        self.min_drop_interval = 150

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_tick)

        # 초기 조각
        if "human" in self.vs_mode:
            self.player_board.spawn_new_piece()
            self.timer.start(self.drop_interval)
        else:
            # RL vs RL: 두 보드 모두 비어있으면 에이전트에서 spawn 하도록 둠
            self.drop_interval = 80  # 좀 더 빠르게
            self.timer.start(self.drop_interval)

        self.update_hold_labels()
        self.update_status()

    # ---------- 자동 틱 ----------

    def game_tick(self):
        if self.player_board.game_over or self.ai_board.game_over:
            self.timer.stop()
            self.update_status()
            return

        if self.vs_mode in ("human_vs_heuristic", "human_vs_rl"):
            # 사람 보드는 중력, AI는 턴 기반
            moved = self.player_board.move(0, 1)
            if not moved:
                cleared = self.player_board.lock_piece()
                self.on_player_piece_locked(cleared)

            self.player_widget.update()
            self.ai_widget.update()
        else:
            # RL vs RL 모드: 양쪽 에이전트가 한 턴씩 둔다 (하드드롭 기반)
            cleared_left = self.rl_left.choose_and_place(self.player_board)
            self.apply_attack_from_player(cleared_left)

            cleared_right = self.rl_right.choose_and_place(self.ai_board)
            self.apply_attack_from_ai(cleared_right)

            self.player_widget.update()
            self.ai_widget.update()

        self.update_status()
        self.update_hold_labels()

    # ---------- 키 입력 ----------

    def keyPressEvent(self, event):
        # RL vs RL 모드에서는 키 입력 무시
        if self.vs_mode == "rl_vs_rl":
            super().keyPressEvent(event)
            return

        if self.player_board.game_over or self.ai_board.game_over:
            super().keyPressEvent(event)
            return

        key = event.key()
        updated = False

        if key in (Qt.Key_Left, Qt.Key_A):
            if self.player_board.move(-1, 0):
                updated = True
        elif key in (Qt.Key_Right, Qt.Key_D):
            if self.player_board.move(1, 0):
                updated = True
        elif key in (Qt.Key_Up, Qt.Key_W):
            if self.player_board.rotate(1):
                updated = True
        elif key in (Qt.Key_Down, Qt.Key_S):
            if self.player_board.move(0, 1):
                updated = True
            else:
                cleared = self.player_board.lock_piece()
                self.on_player_piece_locked(cleared)
                updated = True
        elif key == Qt.Key_Space:
            cleared = self.player_board.hard_drop()
            self.on_player_piece_locked(cleared)
            updated = True
        elif key == Qt.Key_C:
            self.player_board.hold_current_piece()
            updated = True

        if updated:
            self.player_widget.update()
            self.ai_widget.update()
            self.update_hold_labels()
            self.update_status()

        super().keyPressEvent(event)

    # ---------- 사람 턴 끝났을 때 ----------

    def on_player_piece_locked(self, cleared: int):
        self.apply_attack_from_player(cleared)
        self.total_cleared_by_player += cleared
        self.update_speed()

        if not self.player_board.game_over:
            self.player_board.spawn_new_piece()

        self.ai_turn()

    def ai_turn(self):
        if self.ai_board.game_over:
            return
        cleared = self.ai.choose_and_place(self.ai_board)
        self.apply_attack_from_ai(cleared)
        self.ai_widget.update()
        self.update_status()

    # ---------- 속도 조정 (사람 모드에서만) ----------

    def update_speed(self):
        new_level = self.total_cleared_by_player // 10 + 1
        if new_level > self.level:
            self.level = new_level
            self.drop_interval = max(
                self.min_drop_interval, int(self.drop_interval * 0.85)
            )
            self.timer.start(self.drop_interval)
            self.info_label.setText(
                f"Level {self.level} (drop {self.drop_interval}ms), "
                "조작: ←/→ 좌우, ↑ 회전, ↓ 한 칸, Space 하드드롭, C 홀드"
            )

    # ---------- 공격/스코어 ----------

    @staticmethod
    def calc_garbage(cleared: int) -> int:
        return max(0, cleared - 2)

    def apply_attack_from_player(self, cleared: int):
        if cleared > 0:
            self.player_score += cleared
        garbage = self.calc_garbage(cleared)
        if garbage > 0:
            self.ai_board.add_garbage(garbage, easy=True)

    def apply_attack_from_ai(self, cleared: int):
        if cleared > 0:
            self.ai_score += cleared
        garbage = self.calc_garbage(cleared)
        if garbage > 0:
            self.player_board.add_garbage(garbage, easy=True)

    # ---------- UI ----------

    def update_status(self):
        if self.player_board.game_over and self.ai_board.game_over:
            self.status_label.setText(
                f"게임 종료! 둘 다 패배...? (Left {self.player_score} / Right {self.ai_score})"
            )
        elif self.player_board.game_over:
            self.status_label.setText(
                f"게임 종료! 오른쪽 승 (Left {self.player_score} / Right {self.ai_score})"
            )
        elif self.ai_board.game_over:
            self.status_label.setText(
                f"게임 종료! 왼쪽 승 (Left {self.player_score} / Right {self.ai_score})"
            )
        else:
            self.status_label.setText(
                f"Left 점수: {self.player_score} / Right 점수: {self.ai_score}"
            )

    def update_hold_labels(self):
        self.player_hold_label.setText(
            f"Left Hold: {self.player_board.hold_shape or '-'}"
        )
        if self.vs_mode == "rl_vs_rl":
            self.ai_hold_label.setText(
                f"Right Hold: {self.ai_board.hold_shape or '-'} (RL vs RL)"
            )
        else:
            self.ai_hold_label.setText(
                f"Right Hold: {self.ai_board.hold_shape or '-'} "
                f"(모드: {'휴리스틱' if self.vs_mode=='human_vs_heuristic' else 'RL'})"
            )


# =========================
# main
# =========================

def main():
    print("=== Tetris 모드 선택 ===")
    print("1: 사람 vs 휴리스틱 봇")
    print("2: 사람 vs RL 봇 (tetris_policy.pth 필요)")
    print("3: RL vs RL 자동 대전 (tetris_policy.pth 필요)")
    sel = input("번호 입력 (기본=1): ").strip()

    if sel == "2":
        vs_mode = "human_vs_rl"
    elif sel == "3":
        vs_mode = "rl_vs_rl"
    else:
        vs_mode = "human_vs_heuristic"

    app = QApplication(sys.argv)
    w = MainWindow(vs_mode=vs_mode)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
