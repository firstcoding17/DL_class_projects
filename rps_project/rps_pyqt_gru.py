
import sys
import random
import csv
from datetime import datetime
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel
)
from PyQt5.QtCore import Qt

# -----------------------------
# 1. GRU 기반 모델 정의
# -----------------------------
class RPSGRU(nn.Module):
    """
    간단 GRU 모델
    - 입력: (batch, seq_len, 3)  # one-hot(가위/바위/보)
    - 출력: (batch, 3)           # 사람의 다음 수(가위/바위/보)에 대한 로짓
    """
    def __init__(self, input_size=3, hidden_size=16, num_layers=1, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h = self.gru(x)                 # out: (batch, seq_len, hidden)
        last = out[:, -1, :]                 # 마지막 시점만 사용 (batch, hidden)
        logits = self.fc(last)               # (batch, num_classes)
        return logits


# -----------------------------
# 2. AI 래퍼 (데이터 저장 + 학습 + 예측)
# -----------------------------
MOVE_NAMES = ["가위", "바위", "보"]  # 0, 1, 2


class RPSAI:
    def __init__(self, seq_len=5, device="cpu"):
        self.device = torch.device(device)
        self.model = RPSGRU().to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

        self.seq_len = seq_len
        self.history: List[int] = []  # 사람의 수만 기록 (0/1/2)
        self.trained = False

    # --- 유틸 ---
    def _seq_to_tensor(self, seq: List[int]) -> torch.Tensor:
        """
        seq: [0,1,2,...] (길이 L)
        -> (1, L, 3) one-hot 텐서
        """
        L = len(seq)
        x = torch.zeros((1, L, 3), dtype=torch.float32)
        for i, mv in enumerate(seq):
            x[0, i, mv] = 1.0
        return x.to(self.device)

    # --- 재학습 ---
    def train_on_history(self, epochs=20):
        """
        지금까지 history로 다음 수 예측하도록 재학습
        데이터가 작으니까 슬라이딩 윈도우 방식으로 학습
        """
        if len(self.history) < 2:
            return

        self.model.train()

        for ep in range(epochs):
            total_loss = 0.0
            count = 0

            # history: m0, m1, m2, ..., mN
            # i번째 턴을 예측: target = history[i]
            # input은 그 이전 max(seq_len)개의 history
            for i in range(1, len(self.history)):
                start = max(0, i - self.seq_len)
                seq = self.history[start:i]      # 길이 <= seq_len
                x = self._seq_to_tensor(seq)     # (1, L, 3)
                target = torch.tensor(
                    [self.history[i]], dtype=torch.long, device=self.device
                )

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

        self.trained = True

    def _simple_stat_move(self) -> int:
        """
        초반용 간단 통계 전략:
        지금까지 사람이 가장 많이 낸 수를 이기는 수 선택.
        """
        if not self.history:
            return random.randint(0, 2)

        counts = [0, 0, 0]
        for mv in self.history:
            counts[mv] += 1

        most = int(np.argmax(counts))  # 사람이 가장 많이 낸 수
        ai_move = (most + 1) % 3       # 그걸 이기는 수
        return ai_move

    # --- 다음 수 예측 + AI 선택 ---
    def get_ai_move(self) -> int:
        """
        사람의 다음 수를 예측한 후,
        그걸 이기는 패를 AI가 낸다.
        """
        # 데이터가 거의 없거나 아직 학습 안 했으면 간단 통계 기반으로
        if not self.trained or len(self.history) < 5:
            return self._simple_stat_move()

        self.model.eval()
        with torch.no_grad():
            # 최근 seq_len개로 예측
            seq = self.history[-self.seq_len:]
            x = self._seq_to_tensor(seq)       # (1, L, 3)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_user = probs.argmax(dim=1).item()  # 사람이 낼 것 같은 수

        # 사람이 낼 것(pred_user)을 이기는 수 선택
        # 정의: (user + 1) % 3 이 user를 이기는 패(가위=0, 바위=1, 보=2 기준)
        ai_move = (pred_user + 1) % 3
        return ai_move

    def record_user_move(self, user_move: int):
        self.history.append(user_move)


# -----------------------------
# 3. PyQt5 UI
# -----------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GRU 가위바위보 AI 데모")
        self.ai = RPSAI(seq_len=5, device="cpu")

        self.game_count = 0
        self.ai_win = 0
        self.user_win = 0
        self.draw = 0

        # --- 로그 파일 경로 ---
        self.log_path = "rps_logs.csv"
        self._init_log_file()

        # --- 버튼 ---
        self.btn_scissors = QPushButton("가위")
        self.btn_rock = QPushButton("바위")
        self.btn_paper = QPushButton("보")

        self.btn_scissors.clicked.connect(lambda: self.play_round(0))
        self.btn_rock.clicked.connect(lambda: self.play_round(1))
        self.btn_paper.clicked.connect(lambda: self.play_round(2))

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_scissors)
        btn_layout.addWidget(self.btn_rock)
        btn_layout.addWidget(self.btn_paper)

        # --- 결과 표시 라벨 ---
        self.label_info = QLabel("버튼을 눌러 시작하세요.")
        self.label_info.setAlignment(Qt.AlignCenter)

        self.label_last = QLabel("마지막 판: -")
        self.label_last.setAlignment(Qt.AlignCenter)

        self.label_stats = QLabel("전적: 0판 (사용자 0승 / AI 0승 / 무승부 0)")
        self.label_stats.setAlignment(Qt.AlignCenter)

        # --- 레이아웃 ---
        layout = QVBoxLayout()
        layout.addWidget(self.label_info)
        layout.addLayout(btn_layout)
        layout.addWidget(self.label_last)
        layout.addWidget(self.label_stats)

        self.setLayout(layout)
        self.resize(400, 250)

    def _init_log_file(self):
        """로그 파일이 없으면 헤더를 만들어 둔다."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "timestamp", "user_move", "ai_move", "result"])

    def play_round(self, user_move: int):
        """
        한 판 진행:
        1) AI 수 결정
        2) 승패 판정
        3) history에 기록
        4) 10판마다 재학습
        5) 로그 저장
        """
        ai_move = self.ai.get_ai_move()

        # 승패 판정
        # (규칙) ai == (user + 1) % 3 이면 AI 승
        if ai_move == user_move:
            result = "무승부"
            self.draw += 1
        elif ai_move == (user_move + 1) % 3:
            result = "AI 승"
            self.ai_win += 1
        else:
            result = "사용자 승"
            self.user_win += 1

        self.game_count += 1

        # 히스토리 기록 + 재학습 조건 체크
        self.ai.record_user_move(user_move)
        if self.game_count % 10 == 0:
            # 10판마다 지금까지의 history로 재학습
            self.label_info.setText("재학습 중... 잠시만요.")
            QApplication.processEvents()  # UI 멈추지 않게 한 번 갱신
            self.ai.train_on_history(epochs=30)
            self.label_info.setText(f"{self.game_count}판 진행 중 (10판마다 재학습)")

        # 라벨 업데이트
        user_name = MOVE_NAMES[user_move]
        ai_name = MOVE_NAMES[ai_move]

        self.label_last.setText(
            f"마지막 판: 사용자 = {user_name}, AI = {ai_name} → {result}"
        )

        self.label_stats.setText(
            f"전적: {self.game_count}판 "
            f"(사용자 {self.user_win}승 / AI {self.ai_win}승 / 무승부 {self.draw})"
        )

        if self.game_count % 10 != 0:
            self.label_info.setText(f"{self.game_count}판 진행 중 (10판마다 재학습)")

        # --- 로그 저장 ---
        if result == "사용자 승":
            result_tag = "user"
        elif result == "AI 승":
            result_tag = "ai"
        else:
            result_tag = "draw"

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.game_count,
                datetime.now().isoformat(timespec="seconds"),
                user_move,
                ai_move,
                result_tag,
            ])


# -----------------------------
# 4. main
# -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
