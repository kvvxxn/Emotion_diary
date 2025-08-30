import os
import sys
import torch
import numpy as np
import optuna
import random 
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# 절대 경로 설정 -> TEAM_PROJECT 폴더로 이동
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Image_emo 폴더에서 불러오기 - Iamge model setting
from image_emo.utils.label_matching import fix_label
from shared.path import DATA_ROOT, BEST_MODEL_ROOT

# 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# fusion_head 경로 저장용
best_model_path = []

# Image Dataset의 Label에 맞는 Text data를 임의로 가져옴
# Train / Val / Test에 맞춰서 엑셀 파일 분리 해놓음
# Phase = 'train' / 'val' / 'test'에 따라 각 엑셀 파일에서 image_label에 맞는 label의 text를 무작위로 가져올 수 있도록 해야함
# 엑셀 파일: 한글 감정 표현 -> 영어로 바꿔놓음
# image_label은 16개의 Element로 이루어진 벡터이며, 각 Element의 값은 아래 label_to_emotion에서의 숫자값 (0 ~ 6)과 동일
def make_text_dataset(image_label, phase):
    """
    image_label: tensor of shape (batch_size,) containing emotion class indices (0~6)
    phase: 'train', 'val', or 'test'에 따라 다른 엑셀 파일에서 텍스트 추출
    
    반환: text_data (list of sentences) - 배치 사이즈에 맞춰 라벨에 맞는 문장 리스트에서 랜덤 선택 후 리스트 반환
    """
    # 1. 감정 인덱스와 영어 감정명 매핑
    label_to_emotion = {
        0: 'Joy',
        1: 'Sadness',
        2: 'Surprise',
        3: 'Anger',
        4: 'Fear',
        5: 'Disgust',
        6: 'Neutral'
    }
    
    # 2. phase에 따른 파일 경로 지정
    if phase == 'train':
        text_file = os.path.join(DATA_ROOT, 'text_data', 'train_text.xlsx')
    elif phase == 'val':
        text_file = os.path.join(DATA_ROOT, 'text_data', 'val_text.xlsx')
    elif phase == 'test':
        text_file = os.path.join(DATA_ROOT, 'text_data', 'test_text.xlsx')
    else:
        raise ValueError("phase should be one of ['train', 'val', 'test']")
    
    # 3. 엑셀 파일에서 텍스트 데이터 불러오기
    df = pd.read_excel(text_file)
    
    # 4. 감정별 문장들을 딕셔너리로 묶기
    emotion_to_sentences = {}
    for emotion_name in label_to_emotion.values():
        emotion_to_sentences[emotion_name] = df[df['Emotion'] == emotion_name]['Sentence'].tolist()
    
    # 5. image_label에 따른 문장 무작위 선택
    text_data = []
    for label in image_label.cpu().numpy():
        emotion_name = label_to_emotion[label]
        sentences = emotion_to_sentences.get(emotion_name, [""])  # 없으면 빈문장 대체
        if len(sentences) == 0:
            text_data.append("")  # 문장이 없는 경우 빈문장
        else:
            text_data.append(random.choice(sentences))
    
    return text_data


# Optuna: 최적의 Hyperparameter를 자동으로 찾아주는 API
# 2 Layer 같이 얕은 모델에선 그리 느리지않아서 주로 사용한다고 함. 
def objective(
        trial, 
        img_train_loader: DataLoader, 
        img_val_loader: DataLoader, 
        head, 
        img_model, 
        text_model, 
        criterion, 
        tokenizer,
        device: torch.device, 
        num_classes: int, 
        epochs: int,
):
    # 아래 범위 내에서 가장 좋은 Hyperparamter를 탐색함.
    hidden_dim   = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    dropout      = trial.suggest_float("dropout", 0.0, 0.5)
    batchnorm    = trial.suggest_categorical("batchnorm", [False, True])
    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    wd           = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # 2 Layer MLP 세팅
    fc_model = head(hidden_dim, num_classes, dropout, batchnorm).to(device)
    optimizer = torch.optim.Adam(fc_model.parameters(), lr=lr, weight_decay=wd)

    best_acc = 0.0

    # Training
    for epoch in range(1, epochs+1):
        # Image 모델과 Text 모델은 평가 모드로 설정
        img_model.eval()
        text_model.eval()
        # 훈련 모드로 설정
        fc_model.train()

        train_loss = 0.0

        # Image Batch (size: 16)에 맞춰서 진행
        for i, data in enumerate(img_train_loader):
            # Image data['image'] shape: [16, 3, 224, 224]
            img_data = data['image'].to(device)

            # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
            img_labels = data['emotion_label_idx'].to(device)

            # Batch Image에 맞는 7 Class Score Matrix
            targets = fix_label(img_labels).to(device) # (16, )

            text_data = make_text_dataset(targets, 'train')
            
            # 텍스트 데이터 토크나이징
            encoding = tokenizer(
                text_data,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                # 7 element vector가 각각 반환됨
                img_output = img_model(img_data)
                text_output = text_model(input_ids, attention_mask)
            
            # logits 형태로 return
            outputs = fc_model.forward(img_output, text_output)

            optimizer.zero_grad()

            # Target: Dataloader 구성할 때 사용
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.backward()

            optimizer.step()

        train_loss /= (i+1)

        # Validation
        fc_model.eval()
        with torch.no_grad():
            # Validation 정확도 측정
            acc = 0.0
            val_loss = 0.0
                
            for j, data in enumerate(img_val_loader):
                # Image data['image'] shape: [16, 3, 224, 224]
                img_data = data['image'].to(device)

                # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
                img_labels = data['emotion_label_idx'].to(device)

                # Batch Image에 맞는 7 Class Score Matrix
                targets = fix_label(img_labels).to(device) # (16, )

                text_data = make_text_dataset(targets, 'val')
                
                # 텍스트 데이터 토크나이징
                encoding = tokenizer(
                    text_data,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                img_output = img_model(img_data)
                text_output = text_model(input_ids, attention_mask)

                outputs = fc_model.forward(img_output, text_output)

                # Target: Dataloader 구성할 때 사용
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                pred_cls = torch.argmax(outputs, dim = 1)
                mask = (pred_cls == targets).float()
                acc += 100 * (torch.sum(mask).item() / targets.size(0))
            
            # 전체 Batch에 대한 평균값을 계산
            val_loss /= (j + 1)
            acc /= (j + 1)

        if epoch % 5 == 0:
            print("--------------------------------------------------------")
            print(f'Epoch: {epoch}\'s train loss is {train_loss:.4f}')
            print(f'Epoch: {epoch}\'s validation loss is {val_loss:.4f}')
            print(f'Epoch: {epoch}\'s accuracy is {acc:.4f}')
            print("--------------------------------------------------------")
            print()

        # Epoch마다 가장 좋은 모델을 저장하도록 유도
        # 평가 기준은 Accuracy
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned() # Pruning을 통해 빠르게 학습 종료

        # 전체 Epoch에서 Best_acc를 구함
        # best_acc인 경우, Hyperparameter와 Model weight를 저장
        if (acc > best_acc):
            best_acc = acc

            ckpt = {
                "model_state": fc_model.state_dict(),
                "hparams": {"hidden_dim": hidden_dim, "batchnorm": batchnorm, "dropout": dropout},
                "meta": {"trial": int(trial.number), "val_acc": float(acc)}
            }
            path = os.path.join(BEST_MODEL_ROOT, f'fusion_head_model_{trial.number}.pt')
            torch.save(ckpt, path)  
            best_model_path.append(path)

        
    print()
    print(f'Best accuracy across epochs: {best_acc:.4f}')
    print()
    
    return best_acc


def train(
        img_train_loader: DataLoader, 
        img_val_loader: DataLoader,
        mlp, 
        img_model, 
        text_model, 
        criterion, 
        device: torch.device,
        direction: str,
        n_trials: int,
        num_classes: int = 7,
        epochs: int = 10,
):
    best_acc = 0.0

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # direction: maximize
    # best_acc가 최대가 되도록 하는 Hyperparameter를 찾음
    study = optuna.create_study(
        direction=direction,
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=10,  
            # 가장 높은 Trial을 Filtering
            reduction_factor=3
        )
    ) 


    study.optimize(
        lambda trial: objective(
            trial,
            img_train_loader,
            img_val_loader,
            mlp,
            img_model,
            text_model,
            criterion,
            tokenizer,
            device,
            num_classes,
            epochs,
            best_acc
        ),
        n_trials=n_trials
    )

    global_best_acc = 0.0

    for path in best_model_path:
        ckpt = torch.load(path, map_location="cpu")
        acc = ckpt.get("meta", {}).get("val_acc", None)

        if acc > global_best_acc:
            global_best_acc = acc

            new_path = os.path.join(BEST_MODEL_ROOT, f'fusion_head_model.pt')
            torch.save(ckpt, new_path)


    print("Best trial:", study.best_trial.number)
    print("Best value (acc):", study.best_value)
    print("Best params:", study.best_trial.params)

    return study