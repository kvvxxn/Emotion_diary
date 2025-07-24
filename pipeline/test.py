import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vit_model import vit_model
from utils.label_matching import label_to_score


def test(dataloader, device):
    vit, cross_entropy_loss = vit_model()
    vit.load_state_dict(torch.load('best_model.pth'))

    # 평가 모드로 설정
    vit.eval()

    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Image data['image'] shape: [16, 3, 224, 224]
            img = data['image'].to(device)

            # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
            labels = data['emotion_label_idx'].to(device)

            # Batch Image에 맞는 7 Class Score Matrix
            targets = label_to_score(labels).to(device) # (16, )

            # 1. Forward - logits 형태 반환
            outputs = vit(img) # (16, 7)

            # 2. CrossEntropyLoss 내부에서 Softmax 처리 후, Loss 계산 
            loss = cross_entropy_loss(outputs, targets)

            test_loss += loss.item()

            pred_label = torch.argmax(outputs, dim = 1) # 예측 Label
            mask = (pred_label == targets).float()
            test_acc += 100 * torch.sum(mask).item() / targets.size(0)

        # 전체 Batch에 대한 평균값을 계산
        test_loss /= (i + 1)
        test_acc /= (i + 1)
    
    return test_loss, test_acc