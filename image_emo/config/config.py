import os

# root 경로 설정
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'EmoSet-118K')
DATA_ROOT = os.path.abspath(DATA_ROOT) # 절대 경로로 변경

# 원래 EmoSet의 Class 개수
original_num_emotion_classes = 8

# Model이 분류할 Emotion class의 개수
model_num_emotion_classes = 7

# Hyperparameter
num_epoch = 50
learning_rate = [1e-4, 5e-5, 1e-5]
weight_decay = [1e-2, 5e-2]
