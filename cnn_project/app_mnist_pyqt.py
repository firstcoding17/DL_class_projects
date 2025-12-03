# app_mnist_pyqt.py
import sys
import os
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox
)
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint

from model import BetterCNN as SimpleCNN


class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)  # 28x28을 10배 확대해서 그리는 캔버스
        self.image = QImage(self.size(), QImage.Format_Grayscale8)
        self.image.fill(255)  # 흰색 배경
        self.last_point = QPoint()
        self.pen_width = 10  # 선 굵기 (굵을수록 인식 잘 됨)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        """캔버스 지우기"""
        self.image.fill(255)
        self.update()

    def to_28x28_array(self):
        """
        1) QImage(280x280)를 numpy(0~255)로 변환
        2) 색 반전해서 (배경=0, 글씨=1) 방향으로 맞추기
        3) 글씨 있는 영역만 bounding box로 잘라내기
        4) 정사각형으로 패딩
        5) Pillow로 28x28 리사이즈
        6) 0~1 float32 배열 반환
        """
        # 1. QImage -> numpy (원본 280x280)
        w, h = self.image.width(), self.image.height()
        qimg = self.image

        ptr = qimg.bits()
        ptr.setsize(w * h)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w))   # 0~255, 흰 배경 255, 검정 글씨 0

        # 2. 색 반전: 배경(255) -> 0, 글씨(0) -> 1 근처
        arr = 255 - arr
        arr = arr.astype(np.float32) / 255.0   # 0~1

        # 3. 글씨가 있는 부분(bounding box) 찾기
        ys, xs = np.where(arr > 0.1)  # threshold 0.1 정도

        if len(xs) == 0 or len(ys) == 0:
            # 아무것도 안 그렸으면 그냥 28x28 zero 리턴
            return np.zeros((28, 28), dtype=np.float32)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        digit = arr[y_min:y_max+1, x_min:x_max+1]   # 글씨 영역만 잘라냄

        # 4. 정사각형 패딩 (긴 변 기준)
        h_d, w_d = digit.shape
        size = max(h_d, w_d)
        padded = np.zeros((size, size), dtype=np.float32)
        y_offset = (size - h_d) // 2
        x_offset = (size - w_d) // 2
        padded[y_offset:y_offset+h_d, x_offset:x_offset+w_d] = digit

        # 5. Pillow로 28x28 리사이즈 (부드럽게 축소)
        pil_img = Image.fromarray((padded * 255).astype(np.uint8), mode="L")
        pil_img = pil_img.resize((28, 28), resample=Image.BILINEAR)

        arr_28 = np.array(pil_img, dtype=np.float32) / 255.0  # 0~1

        return arr_28


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MNIST 손글씨 숫자 인식 데모")

        # --- 위젯 생성 ---
        self.canvas = DrawingCanvas()
        self.clear_btn = QPushButton("지우기")
        self.predict_btn = QPushButton("예측하기")

        self.result_label = QLabel("예측 결과: -")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18pt;")

        # --- 레이아웃 ---
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # --- 모델 로드 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)

        model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.pth")
        if not os.path.exists(model_path):
            QMessageBox.critical(
                self,
                "에러",
                f"모델 파일({model_path})이 없습니다.\n"
                f"먼저 train_mnist.py를 실행해서 mnist_cnn.pth를 만들어 주세요."
            )
        else:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()

        # --- 시그널 연결 ---
        self.clear_btn.clicked.connect(self.canvas.clear)
        self.predict_btn.clicked.connect(self.predict_digit)

    def predict_digit(self):
        # 모델이 로드되지 않은 경우 대비 (mnist_cnn.pth 없을 때)
        if self.model is None:
            QMessageBox.warning(self, "경고", "모델이 로드되지 않았습니다.")
            return

        # 1) 캔버스 -> 28x28 numpy 배열
        arr = self.canvas.to_28x28_array()  # (28, 28), 값 범위 [0,1]

        # 2) 텐서로 변환: (1, 1, 28, 28)
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(self.device)

 

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        self.result_label.setText(f"예측 결과: {pred} (신뢰도: {confidence*100:.1f}%)")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
