import os
import sys
import torch
import optuna
from torch.utils.data import DataLoader

# 절대 경로 설정 -> TEAM_PROJECT 폴더로 이동
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Image_emo 폴더에서 불러오기 - Iamge model setting
from image_emo.utils.label_matching import fix_label

# Image Dataset의 Label에 맞는 Text data를 임의로 가져옴
# Train / Val / Test에 맞춰서 엑셀 파일 분리 해놓음
# Phase = 'train' / 'val' / 'test'에 따라 각 엑셀 파일에서 image_label에 맞는 label의 text를 무작위로 가져올 수 있도록 해야함
# 엑셀 파일: 한글 감정 표현 -> 영어로 바꿔놓음
# image_label은 16개의 Element로 이루어진 벡터이며, 각 Element의 값은 아래 label_to_emotion에서의 숫자값 (0 ~ 6)과 동일
def make_text_dataset(image_label, phase):
    """
    image_data.shape (16, )
    label_to_emotion = {
        0: 'Joy',
        1: 'Sadness',
        2: 'Surprise',
        3: 'Anger',
        4: 'Fear',
        5: 'Disgust',
        6: 'Neutral'
    }
    """
    

    return text_data


# Optuna: 최적의 Hyperparameter를 자동으로 찾아주는 API
# 2 Layer 같이 얕은 모델에선 그리 느리지않아서 주로 사용한다고 함. 
def objective(model, trial, img_train_loader, img_val_loader, img_model, text_model, criterion, device):
    hidden_dim   = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    dropout      = trial.suggest_float("dropout", 0.0, 0.5)
    batchnorm    = trial.suggest_categorical("batchnorm", [False, True])
    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    wd           = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    epochs       = 30

    model = model(hidden_dim, 7, dropout, batchnorm)
    optimizer = torch.optim.Adam(model.parameters(), lr, wd)
    criterion = criterion

    for i in range(epochs):
        # Training
        model.train()
        for i, data in enumerate(img_train_loader):
            # Image data['image'] shape: [16, 3, 224, 224]
            img_data = data['image'].to(device)

            # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
            img_labels = data['emotion_label_idx'].to(device)

            # Batch Image에 맞는 7 Class Score Matrix
            targets = fix_label(img_labels).to(device) # (16, )

            text_data = make_text_dataset(targets, 'train')

            img_output = img_model(img_data)
            text_output = text_model(text_data)

            output = model.forward(img_data, text_data)

            # Target: Dataloader 구성할 때 사용
            loss = criterion(output, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


        # Validaion
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(img_val_loader):
                # Image data['image'] shape: [16, 3, 224, 224]
                img_data = data['image'].to(device)

                # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
                img_labels = data['emotion_label_idx'].to(device)

                # Batch Image에 맞는 7 Class Score Matrix
                targets = fix_label(img_labels).to(device) # (16, )

                text_data = make_text_dataset(targets, 'val')

                img_output = img_model(img_data)
                text_output = text_model(text_data)

                output = model.forward(img_data, text_data)

                # Target: Dataloader 구성할 때 사용
                loss = criterion(output, targets)

        # Epoch마다 가장 좋은 모델을 저장하도록 유도
        # 평가 기준은 Accuracy

        trial.report(acc, i)
        if trial.should_prune():
            raise optuna.TrialPruned() # Pruning을 통해 빠르게 학습 종료
        

def train(img_train_loader, img_val_loader, mlp, img_model, text_model, criterion, device):
    objective(mlp, trial, img_train_loader, img_val_loader, img_model, text_model, criterion, device)