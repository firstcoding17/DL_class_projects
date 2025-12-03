# 0. 아나콘다 환경 생성
conda create -n dlproject python=3.11
conda activate dlproject

# 1) PyTorch (CPU 버전)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) 나머지
pip install -r requirements.txt

# 3. 데이터생성
python train_mnist.py

# 4. PyQt5 GUI 실행
python app_mnist_pyqt.py
