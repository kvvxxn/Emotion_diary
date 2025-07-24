import torch
from torch.utils.data import DataLoader

from datasets.emoset import EmoSet
from pipeline.train import train_val
from pipeline.test import test
from config.config import data_root, num_emotion_classes


def main():
    # Training dataset
    train_dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=num_emotion_classes,
        phase='train',
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation Dataset
    val_dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=num_emotion_classes,
        phase='val',
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training + Validation
    train_val(train_loader, val_loader, device)

    # Test Dataset
    test_dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=num_emotion_classes,
        phase='test',
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Testing
    test_loss, test_acc = test(test_loader, device)

    print("Average Test Loss: ", test_loss)
    print("Average Test Accuarcy: ", test_acc)

    # To-Do Testing에서의 정답 Label이랑 예측 label 차이를 시각화
    
if __name__ == "__main__":
    main()