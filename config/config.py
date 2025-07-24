import os

# root 경로 설정
data_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'EmoSet-118K')
data_root = os.path.abspath(data_root) # 절대 경로로 변경

# 원래 EmoSet의 Class 개수
num_emotion_classes = 8

# Hyperparameter
num_epoch = [5, 10, 20, 50]
learning_rate = [1e-4, 5e-5, 1e-5]
weight_decay = [1e-2, 5e-2]
