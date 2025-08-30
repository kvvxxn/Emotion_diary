import os

# root 경로 설정
SHARED_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SHARED_ROOT)  # combine 상위
BEST_MODEL_ROOT = os.path.join(SHARED_ROOT, 'best_model')

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_ROOT = os.path.abspath(DATA_ROOT) # 절대 경로로 변경

