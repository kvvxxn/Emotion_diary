import os

# root 경로 설정
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_ROOT = os.path.abspath(DATA_ROOT) # 절대 경로로 변경