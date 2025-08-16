import os
import sys
import torch
import copy
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# 절대 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.models.vit_model import vit_model
from utils.label_matching import fix_label
from config.config import num_epoch, learning_rate, weight_decay

def draw_loss_graph(learning_rate, weight_decay, train_loss_hyper, val_loss_hyper):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    axes = axes.flatten()  # 2차원 → 1차원 flatten

    idx = 0
    for wd in weight_decay:
        for lr in learning_rate:
            key = (lr, wd)
            train_loss = train_loss_hyper.get(key, [])
            val_loss = val_loss_hyper.get(key, [])

            ax = axes[idx]
            ax.plot(train_loss, label='Train Loss', color='red')
            ax.plot(val_loss, label='Validation Loss', color='green', linestyle='--')
            ax.set_title(f"lr={lr}, wd={wd}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
            idx += 1

    # 전체 레이아웃 조정
    plt.suptitle("Train vs Validation Loss for Different Hyperparameters", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def draw_acc_graph(learning_rate, weight_decay, train_acc_hyper, val_acc_hyper):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    axes = axes.flatten()  # 2차원 → 1차원 flatten

    idx = 0
    for wd in weight_decay:
        for lr in learning_rate:
            key = (lr, wd)
            train_acc = train_acc_hyper.get(key, [])
            val_acc = val_acc_hyper.get(key, [])

            ax = axes[idx]
            ax.plot(train_acc, label='Train Accuracy', color='red')
            ax.plot(val_acc, label='Validation Accuracy', color='green', linestyle='--')
            ax.set_title(f"lr={lr}, wd={wd}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.grid(True)
            ax.legend()
            idx += 1

    # 전체 레이아웃 조정
    plt.suptitle("Train vs Validation Accuracy for Different Hyperparameters", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def train(model, criterion, dataloader, optimizer, device):
    # 모델을 훈련 모드로 전환
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, data in enumerate(dataloader):
        # Image data['image'] shape: [16, 3, 224, 224]
        img = data['image'].to(device)

        # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
        labels = data['emotion_label_idx'].to(device)

        # Batch Image에 맞는 7 Class Score Matrix
        targets = fix_label(labels).to(device) # (16, )

        # 1. Forward - logits 형태 반환
        outputs = model(img) # (16, 7)

        # 2. CrossEntropyLoss 내부에서 Softmax 처리 후, Loss 계산 
        loss = criterion(outputs, targets)

        # 3. Backpropagation
        loss.backward()
        optimizer.step()
        # Gradient 초기화
        optimizer.zero_grad()

        train_loss += loss.item()

        idx = torch.argmax(outputs, dim = 1)
        mask = (idx == targets).float()
        train_acc += 100 * (torch.sum(mask).item() / targets.size(0))

    # 전체 Batch에 대한 평균값을 계산
    train_loss /= (i + 1)
    train_acc /= (i + 1)

    return train_loss, train_acc

        

def val(model, criterion, dataloader, device):
    # 모델을 평가 모드로 전환
    model.eval()

    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0

        for i, data in enumerate(dataloader):
            # Image data['image'] shape: [16, 3, 224, 224]
            img = data['image'].to(device)

            # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
            labels = data['emotion_label_idx'].to(device)

            # Batch Image에 맞는 7 Class Score Matrix
            targets = fix_label(labels).to(device) # (16, )

            # 1. Forward - logits 형태 반환
            outputs = model(img) # (16, 7)

            # 2. CrossEntropyLoss 내부에서 Softmax 처리 후, Loss 계산 
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            idx = torch.argmax(outputs, dim = 1)
            mask = (idx == targets).float()
            val_acc += 100 * (torch.sum(mask).item() / targets.size(0))

        # 전체 Batch에 대한 평균값을 계산
        val_loss /= (i + 1)
        val_acc /= (i + 1)

        return val_loss, val_acc 

def train_val(train_loader, val_loader, device):
    # 특정 Hyperparameter (lr, wd) 마다의 Train Loss
    train_loss_hyper = {}  # Key: (lr, wd), Value: loss

    # 특정 Hyperparameter (lr, wd) 마다의 Train Accuracy
    train_acc_hyper = {} # Key: (lr, wd), Value: acc


    # 특정 Hyperparameter (lr, wd) 마다의 Validataion Loss
    val_loss_hyper = {}  # Key: (lr, wd), Value: loss

    # 특정 Hyperparameter (lr, wd) 마다의 Validation Accuracy
    val_acc_hyper = {} # Key: (lr, wd), Value: acc


    for lr in learning_rate:
        for wd in weight_decay:
            # Model Loading
            # Hyperparameter가 바뀔 때마다 초기화되어야 됨
            vit, cross_entropy_loss = vit_model()

            # learning rate, weight decay에 맞추어 optimizer 설정
            optimizer = optim.Adam(vit.parameters(), lr=lr, weight_decay=wd)

            # Epoch마다의 Train Loss를 저장
            train_loss_per_epoch = []
            # Epoch마다의 Train Accuracy를 저장
            train_acc_per_epoch = []

            # Epoch마다의 Validation Loss를 저장
            val_loss_per_epoch = []
            # Epoch마다의 Validation Accuracy를 저장
            val_acc_per_epoch = []
            
            # 가장 좋은 성능의 Loss, Model, Hyperparameter를 저장
            best_loss = float('inf')
            best_hyperparam = None # tuple: (lr, wd, epoch)
            best_model = None

            for epoch in range(num_epoch):
                # learning_rate, Weight_dacay, Epoch에 맞춰 모델 학습
                train_loss, train_acc = train(vit, cross_entropy_loss, train_loader, optimizer, device)

                # learning_rate, Weight_dacay, Epoch에 따른 모델의 성능 평가
                val_loss, val_acc = val(vit, cross_entropy_loss, val_loader, device)

                train_loss_per_epoch.append(train_loss) # list
                train_acc_per_epoch.append(train_acc) # list

                val_loss_per_epoch.append(val_loss) # list
                val_acc_per_epoch.append(val_acc) # list

                if (best_loss > val_loss):
                    best_loss = val_loss
                    best_hyperparam = (lr, wd, epoch)
                    best_model = copy.deepcopy(vit)

            train_loss_hyper[(lr, wd)] = train_loss_per_epoch
            train_acc_hyper[(lr, wd)] = train_acc_per_epoch

            val_loss_hyper[(lr, wd)] = val_loss_per_epoch
            val_acc_hyper[(lr, wd)] = val_acc_per_epoch

    # 가장 좋은 결과의 모델을 파일에 저장
    torch.save(best_model.state_dict(), "best_model.pth")

    # 2행 3열의 Loss graph 그리기
    draw_loss_graph(learning_rate, weight_decay, train_loss_hyper, val_loss_hyper)

    # 2행 3열의 Accuracy 그래프 그리기
    draw_acc_graph(learning_rate, weight_decay, train_acc_hyper, val_acc_hyper)

    # Debugging
    print("----------Best Loss----------")
    print(best_loss)
    print()

    print("----------Best Hyperparameter----------")
    print(best_hyperparam)
    print()

    print("----------Minimum loss in each Hyperparameter----------")
    for key in train_loss_hyper:
        print(f'Training loss in ({key[0], key[1]}): {np.min(train_loss_hyper[key])}')
        print(f'Validation loss in ({key[0], key[1]}): {np.min(val_loss_hyper[key])}')
        print()

    print("----------Maximum accuracy in each Hyperparameter----------")
    for key in train_acc_hyper:
        print(f'Training accuracy in ({key[0], key[1]}): {np.max(train_acc_hyper[key])}')
        print(f'Validation accuracy in ({key[0], key[1]}): {np.max(val_acc_hyper[key])}')
        print()

    return best_hyperparam