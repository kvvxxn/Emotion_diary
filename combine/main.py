import os
import sys
import torch
from torch.utils.data import DataLoader

from pipeline.train import train
from pipeline.test import test

# 절대 경로 설정 -> TEAM_PROJECT 폴더로 이동
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Image_emo 폴더에서 불러오기 - Iamge model setting
from image_emo.config.config import original_num_emotion_classes

# Shared 폴더에서 불러오기 - 모델, Dataset
from shared.datasets.image_emoset import EmoSet
from shared.models.fusion_head import FusionMLP
from shared.models.vit_model import vit_model
from shared.models.text_emotion_classifier import EmotionClassifier
from shared.path import DATA_ROOT

# shared/data 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # combine 상위
SHARED_ROOT = os.path.join(PROJECT_ROOT, 'shared')

# Image Data, best model 경로 설정
IMAGE_DATA_ROOT = os.path.join(DATA_ROOT,  'image_data', 'EmoSet-118K')
IMAGE_BEST_MODEL_PATH = os.path.join(SHARED_ROOT, 'best_model', 'image_best_model.pth')
TEXT_BEST_MODEL_PATH = os.path.join(SHARED_ROOT, 'best_model', 'text_best_model.pt')

# Text Data 경로 설정
TRAIN_TEXT_PATH = os.path.join(DATA_ROOT, 'text_data', 'train_text.xlsx')
VAL_TEXT_PATH = os.path.join(DATA_ROOT, 'text_data', 'val_text.xlsx')
TEST_TEXT_PATH = os.path.join(DATA_ROOT, 'text_data', 'test_text.xlsx')

# Device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image Model 불러오기
image_model_path = os.path.join('models', 'image_best_model.pth')
vit, cross_entropy_loss = vit_model()
vit.load_dict(torch.load(IMAGE_BEST_MODEL_PATH), required_grad = False)

# Image Dataset 불러오기
img_train_dataset = EmoSet(
    data_root=IMAGE_DATA_ROOT,
    num_emotion_classes=original_num_emotion_classes,
    phase='train',
)
img_train_loader = DataLoader(img_train_dataset, batch_size=16, shuffle=True)

# Validation Dataset
img_val_dataset = EmoSet(
    data_root=IMAGE_DATA_ROOT,
    num_emotion_classes=original_num_emotion_classes,
    phase='val',
)
img_val_loader = DataLoader(img_val_dataset, batch_size=16, shuffle=False)

# Test Dataset
img_test_dataset = EmoSet(
    data_root=IMAGE_DATA_ROOT,
    num_emotion_classes=original_num_emotion_classes,
    phase='test',
)
img_test_loader = DataLoader(img_test_dataset, batch_size=16, shuffle=False)


######################################################
# TODO: Text Model 불러오기 / 변수명: 또는 bert
bert = EmotionClassifier()
bert.load_state_dict(torch.load(TEXT_BEST_MODEL_PATH))
bert.to(device)
bert.eval()
######################################################


# FusionMLP를 Training + Validation
train(img_train_loader, img_val_loader, vit, bert, cross_entropy_loss, device)
test(img_test_loader, vit, bert)