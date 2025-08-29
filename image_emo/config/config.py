import os

# 원래 EmoSet의 Class 개수
original_num_emotion_classes = 8

# Model이 분류할 Emotion class의 개수
model_num_emotion_classes = 7

# Hyperparameter
num_epoch = 50
learning_rate = [1e-4, 5e-5, 1e-5]
weight_decay = [1e-2, 5e-2]
