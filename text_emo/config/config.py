import os

# 데이터 경로
#data_root = os.path.join("data", "한국어 감정 정보가 포함된 단발성 대화 데이터셋")
#data_file = os.path.join(data_root, "한국어_단발성_대화_데이터셋.xlsx")
file_path = "/content/drive/MyDrive/Project/KOR/data/한국어 감정 정보가 포함된 단발성 대화 데이터셋/한국어_단발성_대화_데이터셋.xlsx"

# 감정 라벨
emotion_list = ['행복', '슬픔', '놀람', '분노', '공포', '혐오', '중립']

# 하이퍼파라미터
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = [1e-4, 5e-5, 2e-5]
WEIGHT_DECAY = [1e-2, 1e-3, 5e-4]

#MODEL_SAVE_PATH = ""
MODEL_SAVE_PATH = f"/content/drive/MyDrive/Project/KOR/model/"