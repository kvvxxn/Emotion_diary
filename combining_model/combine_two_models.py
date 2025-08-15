import os
import sys
import torch
import optuna
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 절대 경로 설정 -> TEAM_PROJECT 폴더로 이동
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_emo.datasets.emoset import EmoSet
from image_emo.models.vit_model import vit_model
from image_emo.config.config import original_num_emotion_classes, data_root
from image_emo.utils.label_matching import label_to_score

# Image Model 불러오기
image_model_path = os.path.join('models', 'image_best_model.pth')
vit, cross_entropy_loss = vit_model()
vit.load_dict(torch.load(image_model_path), required_grad = False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Text Model 불러오기 / 변수명: text_model 또는 bert

class MLP(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.5, batchnorm: bool = False):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(14) if batchnorm else nn.Identity()
        self.fc1 = torch.nn.Linear(14, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    
    def forward(self, img_vec: torch.Tensor, text_vec: torch.Tensor):
        # 두 벡터 연결
        input = torch.concatenate([img_vec, text_vec], dim=1)

        input = self.bn1(input)
        hidden = self.fc1(input)
        hidden = self.bn2(hidden)
        hidden = F.relu(hidden, inplace=True)

        output = self.fc2(hidden)

        return output
    
img_train_dataset = EmoSet(
    data_root=data_root,
    num_emotion_classes=original_num_emotion_classes,
    phase='train',
)
img_train_loader = DataLoader(img_train_dataset, batch_size=16, shuffle=True)

# Validation Dataset
img_val_dataset = EmoSet(
    data_root=data_root,
    num_emotion_classes=original_num_emotion_classes,
    phase='val',
)
img_val_loader = DataLoader(img_val_dataset, batch_size=16, shuffle=False)

# Test Dataset
img_test_dataset = EmoSet(
    data_root=data_root,
    num_emotion_classes=original_num_emotion_classes,
    phase='test',
)
img_test_loader = DataLoader(img_test_dataset, batch_size=16, shuffle=False)
    
# Image Dataset의 Label에 맞는 Text data를 임의로 가져옴
def make_text_dataset(image_label, phase):
    """
    image_data.shape (16, )
    """
    # Training dataset
    


    vit.eval()

    img_vec = vit()
    text_vec = bert()

    fusion_mlp = MLP()

    # Train_loader에 text, img 모두 포함되어야 함
    return text_data

    

def objective(trial, train_loader, val_loader):
    hidden_dim   = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    dropout      = trial.suggest_float("dropout", 0.0, 0.5)
    batchnorm    = trial.suggest_categorical("batchnorm", [False, True])
    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    wd           = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    epochs       = 30

    model = MLP(hidden_dim, 7, dropout, batchnorm)
    optimizer = torch.optim.Adam(model.parameters(), lr, wd)
    criterion = cross_entropy_loss

    for i in range(epochs):
        # Training
        model.train()
        for i, data in enumerate(img_train_loader):
            # Image data['image'] shape: [16, 3, 224, 224]
            img_data = data['image'].to(device)

            # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
            img_labels = data['emotion_label_idx'].to(device)

            # Batch Image에 맞는 7 Class Score Matrix
            targets = label_to_score(img_labels).to(device) # (16, )

            text_data = make_text_dataset(targets, 'train')

            img_output = vit(img_data)
            text_output = bert(text_data)

            output = model.forward(img_data, text_data)

            # Target: Dataloader 구성할 때 사용
            loss = criterion(output, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


        # Validaion
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # Image data['image'] shape: [16, 3, 224, 224]
                img_data = data['image'].to(device)

                # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
                img_labels = data['emotion_label_idx'].to(device)

                # Batch Image에 맞는 7 Class Score Matrix
                targets = label_to_score(img_labels).to(device) # (16, )

                text_data = make_text_dataset(targets, 'val')

                img_output = vit(img_data)
                text_output = bert(text_data)

                output = model.forward(img_data, text_data)

                # Target: Dataloader 구성할 때 사용
                loss = criterion(output, targets)

        # Epoch마다 가장 좋은 모델을 저장하도록 유도
        # 평가 기준은 Accuracy

        trial.report(acc, i)
        if trial.should_prune():
            raise optuna.TrialPruned() # Pruning을 통해 빠르게 학습 종료
    

def training_mlp():
    # TODO: 같은 Label의 이미지, 텍스트 데이터셋 구축 필요
    
    vit.eval()

    with torch.no_grad():
        img_vec = vit() #벡터만
        text_vec = bert()

    # TODO: Text Model 돌리고 text_vec 얻기

    # 
    
    return emo_vec

def emo_predict():
     # TODO: 이미지, 텍스트 입력 받기
    input_image = input()

    # 베스트 Fusion Model을 불러옴

