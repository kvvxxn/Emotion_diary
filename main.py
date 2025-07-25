import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from datasets.emoset import EmoSet
from pipeline.train import train_val
from pipeline.test import test
from config.config import data_root, original_num_emotion_classes

def set_seed(seed=42):
    # Random seed 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # GPU에 대해 Random seed 설정
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()

    # Training dataset
    train_dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=original_num_emotion_classes,
        phase='train',
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation Dataset
    val_dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=original_num_emotion_classes,
        phase='val',
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training + Validation
    # Return type: Tuple: (lr, wd, epoch)
    best_hyperparams = train_val(train_loader, val_loader, device)

    print("-----Best Hyperparameter-----")
    print("Learning rate:", best_hyperparams[0])
    print("Weight Decay:", best_hyperparams[1])
    print("Epochs:", best_hyperparams[2])
    print()

    # Test Dataset
    test_dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=original_num_emotion_classes,
        phase='test',
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Testing
    test_loss, test_acc = test(test_loader, device, visualize=True, num_visualize=10)

    print("Average Test Loss: ", test_loss)
    print("Average Test Accuarcy: ", test_acc)

    # To-Do Testing에서의 정답 Label이랑 예측 label 차이를 시각화
    
if __name__ == "__main__":
    main()