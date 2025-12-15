# 0. 아나콘다 환경 생성
conda create -n dlproject python=3.11
conda activate dlproject

# 1) PyTorch (CPU 버전)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) 나머지
pip install -r requirements.txt

# 3. 데이터생성 -cnn_project
cd cnn_project
python train_mnist.py

# 4. PyQt5 GUI 실행
python app_mnist_pyqt.py

#build rps_qru
cd rps_project
python rps_pyqt_gru.py

#build tetris

cd Tetris

#run Tetris
python tetris_pyqt_ai.py
# 1 휴리스틱
# 2 강화학습
# 3 강화학습 vs 강화학습

#train tetris ai 에피소드, 환경 설정 가능
python train_tetris_vs_heuristic_parallel.py --episodes 20000 --num_envs 20
