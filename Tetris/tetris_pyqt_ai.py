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
    "L": QColor(255, 127, 0),
}


# =========================
# 테트리스 보드
# =========================

class TetrisBoard:
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height

        self.board = [["" for _ in range(self.width)] for _ in range(self.height)]

        self.current_shape = ""
        self.current_rotation = 0
        self.current_x = 0
        self.current_y = 0

        self.hold_shape = None
        self.can_hold = True

        self.game_over = False

    # ---------- 유틸 ----------

    def clone_board_matrix(self):
        return copy.deepcopy(self.board)

    @staticmethod
    def get_shape_cells_static(shape: str, rotation: int, x: int, y: int):
        cells = []
        for dx, dy in TETROMINOES[shape][rotation]:
            cells.append((x + dx, y + dy))
        return cells

    @staticmethod
    def is_valid_position_static(board, shape, rotation, x, y):
        height = len(board)
        width = len(board[0])
        for dx, dy in TETROMINOES[shape][rotation]:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                return False
            if board[ny][nx] != "":
                return False
        return True

    def is_valid_position(self, shape, rotation, x, y) -> bool:
        return TetrisBoard.is_valid_position_static(self.board, shape, rotation, x, y)

    # ---------- 스폰 ----------

    def spawn_new_piece(self):
        self.current_shape = random.choice(SHAPES)
        self.current_rotation = 0
        self.current_x = self.width // 2 - 2
        self.current_y = 0

        if not self.is_valid_position(
            self.current_shape, self.current_rotation, self.current_x, self.current_y
        ):
            self.game_over = True

    # ---------- 이동/회전 ----------

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

    def rotate(self, dr: int) -> bool:
        if not self.current_shape:
            return False
        new_rot = (self.current_rotation + dr) % len(TETROMINOES[self.current_shape])
        if self.is_valid_position(self.current_shape, new_rot, self.current_x, self.current_y):
            self.current_rotation = new_rot
            return True
        return False

    # ---------- 하드 드롭 ----------

    def hard_drop(self) -> int:
        if not self.current_shape:
            return 0
        while self.move(0, 1):
            pass
        return self.lock_piece()

    # ---------- 조각 고정 & 줄 제거 ----------

    def lock_piece(self) -> int:
        if not self.current_shape:
            return 0

        for dx, dy in TETROMINOES[self.current_shape][self.current_rotation]:
            nx = self.current_x + dx
            ny = self.current_y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.board[ny][nx] = self.current_shape

        cleared = self.clear_lines()

        self.current_shape = ""
        self.current_rotation = 0
        self.current_x = 0
        self.current_y = 0
        self.can_hold = True

        if not self.game_over:
            self.spawn_new_piece()

        return cleared

    def clear_lines(self) -> int:
        new_board = []
        cleared = 0
        for row in self.board:
            if all(cell != "" for cell in row):
                cleared += 1
            else:
                new_board.append(row)
        while len(new_board) < self.height:
            new_board.insert(0, ["" for _ in range(self.width)])
        self.board = new_board
        return cleared

    # ---------- 홀드 ----------

    def hold_current_piece(self):
        if not self.can_hold or not self.current_shape:
            return

        if self.hold_shape is None:
            self.hold_shape = self.current_shape
            self.spawn_new_piece()
        else:
            temp = self.current_shape
            new_shape = self.hold_shape
            start_x = self.width // 2 - 2
            start_y = 0
            if self.is_valid_position(new_shape, 0, start_x, start_y):
                self.hold_shape = temp
                self.current_shape = new_shape
                self.current_rotation = 0
                self.current_x = start_x
                self.current_y = start_y
            else:
                self.game_over = True

        self.can_hold = False

    # ---------- 쓰레기 줄(공격) ----------

    def add_garbage(self, n_lines: int, easy: bool = True):
        if n_lines <= 0:
            return
        for _ in range(n_lines):
            # 맨 위 제거
            self.board.pop(0)
            # 새 줄 추가
            row = ["#" for _ in range(self.width)]
            if easy:
                hole = random.randint(0, self.width - 1)
                row[hole] = ""
            self.board.append(row)

        # 현재 조각이 겹쳐서 게임오버인지 체크
        if self.current_shape:
            if not self.is_valid_position(
                self.current_shape, self.current_rotation, self.current_x, self.current_y
            ):
                self.game_over = True

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

        # 줄 제거 가중치 주고, 높이/구멍/울퉁불퉁 패널티
        score = (
            1.5 * lines_cleared
            - 0.5 * aggregate_height
            - 0.7 * holes
            - 0.3 * bumpiness
        )
        return score


# =========================
# 휴리스틱 AI
# =========================

class TetrisAI:
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height

    @staticmethod
    def simulate_drop_and_lock(board: List[List[str]], shape: str, rotation: int, x: int, height: int):
        y = 0
        while TetrisBoard.is_valid_position_static(board, shape, rotation, x, y + 1):
            y += 1
        for dx, dy in TETROMINOES[shape][rotation]:
            ny = y + dy
            nx = x + dx
            if 0 <= nx < len(board[0]) and 0 <= ny < height:
                board[ny][nx] = shape
        new_rows, cleared = TetrisAI.sim_clear_lines(board)
        return new_rows, cleared

    @staticmethod
    def sim_clear_lines(board: List[List[str]]):
        new_board = []
        cleared = 0
        for row in board:
            if all(cell != "" for cell in row):
                cleared += 1
            else:
                new_board.append(row)
        while len(new_board) < len(board):
            new_board.insert(0, ["" for _ in range(len(board[0]))])
        return new_board, cleared

    def choose_and_place(self, board: TetrisBoard) -> int:
        if not board.current_shape:
            board.spawn_new_piece()
            if board.game_over:
                return 0

        shape = board.current_shape
        best_score = -1e9
        best_x = board.current_x
        best_rot = board.current_rotation

        for rot_idx in range(len(TETROMINOES[shape])):
            for x in range(-2, board.width):
                if not TetrisBoard.is_valid_position_static(
                    board.board, shape, rot_idx, x, 0
                ):
                    continue

                sim_bd = copy.deepcopy(board.board)
                sim_bd, lines_cleared = self.simulate_drop_and_lock(
                    sim_bd, shape, rot_idx, x, board.height
                )
                score = TetrisBoard.evaluate_board(sim_bd, lines_cleared)

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_rot = rot_idx

        board.current_rotation = best_rot
        board.current_x = best_x
        board.current_y = 0

        while board.move(0, 1):
            pass
        cleared = board.lock_piece()
        return cleared


# =========================
# RL Policy (train_tetris_rl.py와 동일 구조)
# =========================

class PolicyNet(nn.Module):
    """
    state_dim = 214 (보드 200 + 현재조각 7 + 홀드 7)
    n_actions = 41 (10*4 + hold)
    """
    def __init__(self, state_dim=214, n_actions=41, hidden=256):
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
    RL 정책으로 (rot, x) 또는 hold를 선택해서 한 번에 수를 두는 봇.
    - action in [0 .. width*4-1] : 배치
    - action == width*4          : hold
    """

    def __init__(self, board_width: int,
                 model_path: str = "tetris_vs_heuristic_hold_policy_parallel.pth"):
        self.width = board_width
        # state_dim = 214 (board 200 + current 7 + hold 7)
        self.state_dim = 10 * 20 + len(SHAPES) + len(SHAPES)
        # n_actions = 41 (10*4 + 1 hold)
        self.n_actions = self.width * 4 + 1
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

        # RL이 이상하게 행동하면 fallback으로 쓸 휴리스틱
        self.backup_ai = TetrisAI()
        # 연속 hold 감지용
        self.hold_counter = 0

    def _make_state(self, board: TetrisBoard) -> np.ndarray:
        """
        PyQt용 TetrisBoard → RL state (214차원)
        """
        h = board.height
        w = board.width
        mat = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                if board.board[y][x] != "":
                    mat[y, x] = 1.0
        flat = mat.reshape(-1)  # 200

        cur_onehot = np.zeros(len(SHAPES), dtype=np.float32)
        if board.current_shape:
            cur_idx = SHAPES.index(board.current_shape)
            cur_onehot[cur_idx] = 1.0

        hold_onehot = np.zeros(len(SHAPES), dtype=np.float32)
        if board.hold_shape:
            hold_idx = SHAPES.index(board.hold_shape)
            hold_onehot[hold_idx] = 1.0

        state = np.concatenate([flat, cur_onehot, hold_onehot], axis=0)
        return state

    def choose_and_place(self, board: TetrisBoard) -> int:
        """
        한 턴에 RL이 할 행동:
        - hold를 선택하면 hold만 하고 줄 제거는 없음 (0 리턴)
        - 배치를 선택하면 하드드롭 + 제거한 줄 수 리턴
        """
        # 모델이 없으면 휴리스틱 사용
        if self.policy is None:
            return self.backup_ai.choose_and_place(board)

        if board.game_over:
            return 0

        if not board.current_shape:
            board.spawn_new_piece()
            if board.game_over:
                return 0

        # 상태 만들기
        state = self._make_state(board)
        s_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)

        # 액션 고르기
        with torch.no_grad():
            logits = self.policy(s_tensor)
            action = torch.argmax(logits, dim=1).item()

        print(f"[RL] action={action}/{self.n_actions - 1}  (hold={action == self.n_actions - 1})")

        # -------------------------
        # 1) HOLD 액션 처리
        # -------------------------
        if action == self.n_actions - 1:
            # 이미 홀드 못 쓰는 상태면 → 휴리스틱으로 한 수 둠
            if not board.can_hold:
                print("[RL] hold 선택했지만 can_hold=False → 휴리스틱 fallback")
                return self.backup_ai.choose_and_place(board)

            self.hold_counter += 1
            print(f"[RL] hold_counter={self.hold_counter}")

            # 연속으로 hold만 누르면 재미없으니 N번 이상이면 휴리스틱으로 보정
            if self.hold_counter >= 3:
                print("[RL] hold 연타 → 휴리스틱 fallback으로 한 수 둠")
                self.hold_counter = 0
                return self.backup_ai.choose_and_place(board)

            board.hold_current_piece()
            # hold는 줄 제거 없음
            return 0

        # HOLD가 아닌 배치면 카운터 리셋
        self.hold_counter = 0

        # -------------------------
        # 2) 배치 액션 처리
        # -------------------------
        rot_id = action // self.width
        x = action % self.width

        shape = board.current_shape
        num_rots = len(TETROMINOES[shape])
        rotation = rot_id % num_rots

        # (중요) candidate 위치 (x, 0) 기준으로 유효성 체크
        if not board.is_valid_position(shape, rotation, x, 0):
            print(f"[RL] invalid placement: shape={shape}, x={x}, rot={rotation} → 휴리스틱 fallback")
            return self.backup_ai.choose_and_place(board)

        board.current_rotation = rotation
        board.current_x = x
        board.current_y = 0

        cleared = board.hard_drop()
        print(f"[RL] placed: shape={shape}, x={x}, rot={rotation}, cleared={cleared}")
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

        # 그리드 & 고정 블럭
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

        # 현재 떨어지는 조각
        if self.board.current_shape:
            for dx, dy in TETROMINOES[self.board.current_shape][self.board.current_rotation]:
                bx = self.board.current_x + dx
                by = self.board.current_y + dy
                if 0 <= bx < self.board.width and 0 <= by < self.board.height:
                    rect_x = bx * self.cell_size
                    rect_y = by * self.cell_size
                    color = COLOR_MAP.get(self.board.current_shape, QColor(255, 255, 255))
                    painter.fillRect(
                        rect_x + 1,
                        rect_y + 1,
                        self.cell_size - 1,
                        self.cell_size - 1,
                        color,
                    )


# =========================
# 메인 윈도우
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

        self.ai = None
        self.rl_left = None
        self.rl_right = None

        if vs_mode == "human_vs_heuristic":
            self.ai = TetrisAI(self.ai_board.width, self.ai_board.height)
        elif vs_mode == "human_vs_rl":
            self.ai = RLTetrisAI(self.ai_board.width, model_path="tetris_vs_heuristic_hold_policy.pth")
        else:
            self.rl_left = RLTetrisAI(self.player_board.width, model_path="tetris_vs_heuristic_hold_policy.pth")
            self.rl_right = RLTetrisAI(self.ai_board.width, model_path="tetris_vs_heuristic_hold_policy.pth")

        self.player_widget = BoardWidget(self.player_board)
        self.ai_widget = BoardWidget(self.ai_board)

        if "human" in vs_mode:
            info_text = "조작: ←/→ 좌우, ↑ 회전, ↓ 한 칸, Space 하드드롭, C 홀드"
        else:
            info_text = "RL vs RL 모드입니다 (조작 없음)."

        self.info_label = QLabel(info_text)
        self.info_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.player_hold_label = QLabel("Left Hold: -")
        self.player_hold_label.setAlignment(Qt.AlignCenter)
        self.ai_hold_label = QLabel("Right Hold: -")
        self.ai_hold_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        boards_layout = QHBoxLayout()
        boards_layout.addWidget(self.player_widget)
        boards_layout.addWidget(self.ai_widget)

        layout.addLayout(boards_layout)
        layout.addWidget(self.info_label)
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

        # 콤보 카운터 (연속 줄 제거 시 추가 공격용)
        self.combo_player = 0
        self.combo_ai = 0

        # 락 딜레이: 블록이 바닥/다른 블록에 닿은 뒤 고정까지 대기 시간(ms)
        self.lock_delay_ms = 300  # 0.2~0.5초 정도에서 조정해도 됨
        self.lock_timer = QTimer(self)
        self.lock_timer.setSingleShot(True)
        self.lock_timer.timeout.connect(self.lock_player_piece_delayed)

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
                # 더 내려갈 수 없으면 락 딜레이 타이머 시작
                if not self.lock_timer.isActive():
                    self.lock_timer.start(self.lock_delay_ms)

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
                # 움직였으면 락 딜레이 취소
                if self.lock_timer.isActive():
                    self.lock_timer.stop()
                updated = True
        elif key in (Qt.Key_Right, Qt.Key_D):
            if self.player_board.move(1, 0):
                if self.lock_timer.isActive():
                    self.lock_timer.stop()
                updated = True
        elif key in (Qt.Key_Up, Qt.Key_W):
            if self.player_board.rotate(1):
                if self.lock_timer.isActive():
                    self.lock_timer.stop()
                updated = True
        elif key in (Qt.Key_Down, Qt.Key_S):
            if self.player_board.move(0, 1):
                # 한 칸 내려가면 더 이상 바닥이 아닐 수 있으므로 락 딜레이 취소
                if self.lock_timer.isActive():
                    self.lock_timer.stop()
                updated = True
            else:
                # 더 내려갈 수 없으면 락 딜레이 타이머 시작
                if not self.lock_timer.isActive():
                    self.lock_timer.start(self.lock_delay_ms)
                updated = True
        elif key == Qt.Key_Space:
            # 하드드롭은 즉시 고정 -> 락 딜레이 취소 후 바로 고정
            if self.lock_timer.isActive():
                self.lock_timer.stop()
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

        # 사람 한 턴 끝나면 AI도 한 수 둠
        self.ai_turn()

    def ai_turn(self):
        if self.vs_mode == "human_vs_heuristic":
            if not self.ai_board.current_shape:
                self.ai_board.spawn_new_piece()
            if not self.ai_board.game_over:
                cleared = self.ai.choose_and_place(self.ai_board)
                self.apply_attack_from_ai(cleared)
        elif self.vs_mode == "human_vs_rl":
            if not self.ai_board.current_shape:
                self.ai_board.spawn_new_piece()
            if not self.ai_board.game_over:
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
        # 한 번에 3줄 이상 지웠을 때만 공격 (클래식 룰)
        return max(0, cleared - 2)

    def apply_attack_from_player(self, cleared: int):
        # 플레이어 줄 제거 및 콤보/공격 계산
        if cleared > 0:
            self.player_score += cleared
            self.combo_player += 1
        else:
            self.combo_player = 0

        base = self.calc_garbage(cleared)
        combo_bonus = max(0, self.combo_player - 1)
        garbage = base + combo_bonus

        if garbage > 0:
            self.ai_board.add_garbage(garbage, easy=True)

    def apply_attack_from_ai(self, cleared: int):
        # AI 줄 제거 및 콤보/공격 계산
        if cleared > 0:
            self.ai_score += cleared
            self.combo_ai += 1
        else:
            self.combo_ai = 0

        base = self.calc_garbage(cleared)
        combo_bonus = max(0, self.combo_ai - 1)
        garbage = base + combo_bonus

        if garbage > 0:
            self.player_board.add_garbage(garbage, easy=True)

    # ---------- 락 딜레이 타이머 콜백 ----------

    def lock_player_piece_delayed(self):
        """락 딜레이 만료 시 사람 조각을 실제로 고정하는 함수."""
        # 사람 모드가 아니면 무시
        if "human" not in self.vs_mode:
            return

        # 이미 게임이 끝났으면 무시
        if self.player_board.game_over:
            return

        # 현재 조각이 없으면(중간에 하드드롭/홀드 등으로 상태가 바뀐 경우) 무시
        if not self.player_board.current_shape:
            return

        cleared = self.player_board.lock_piece()
        self.on_player_piece_locked(cleared)
        self.player_widget.update()
        self.update_hold_labels()
        self.update_status()

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
        p_hold = self.player_board.hold_shape or "-"
        a_hold = self.ai_board.hold_shape or "-"
        self.player_hold_label.setText(f"Left Hold: {p_hold}")
        self.ai_hold_label.setText(f"Right Hold: {a_hold}")


# =========================
# main
# =========================

def main():
    print("=== Tetris 모드 선택 ===")
    print("1: 사람 vs 휴리스틱 봇")
    print("2: 사람 vs RL 봇 (tetris_vs_heuristic_hold_policy.pth 필요)")
    print("3: RL vs RL 자동 대전 (tetris_vs_heuristic_hold_policy.pth 필요)")
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
