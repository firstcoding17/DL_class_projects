# GRU 기반 가위바위보 AI (PyQt5)

## 1. 개요

- PyQt5 버튼으로 사람이 **가위/바위/보**를 선택하면,
- AI가 GRU 모델을 이용해 **사람의 다음 수를 예측**하고,
- 그 수를 이길 수 있는 패를 선택하는 가위바위보 게임입니다.
- 10판마다 지금까지의 기록(history)을 사용해 **온라인 재학습**을 수행합니다.
- 모든 게임 기록은 `rps_logs.csv`로 저장됩니다.

## 2. 환경 설정

```bash
conda activate dlproject
pip install torch torchvision PyQt5 numpy Pillow
