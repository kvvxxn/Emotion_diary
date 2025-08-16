import os
import sys
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.models.vit_model import vit_model
from utils.label_matching import fix_label


def test(dataloader, device, visualize=False, num_visualize=6):
    vit, cross_entropy_loss = vit_model()
    vit.load_state_dict(torch.load('best_model.pth')) # main.py에서 실행했을 때 기준 파일

    # 평가 모드로 설정
    vit.eval()

    test_loss = 0.0
    test_acc = 0.0

    visual_imgs = []
    visual_preds = []
    visual_targets = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Image data['image'] shape: [16, 3, 224, 224]
            img = data['image'].to(device)

            # Label data['emotion_label_idx'] shape: (16, ), dtype: int64
            labels = data['emotion_label_idx'].to(device)

            # Batch Image에 맞는 7 Class Score Matrix
            targets = fix_label(labels).to(device) # (16, )

            # 1. Forward - logits 형태 반환
            outputs = vit(img) # (16, 7)

            # 2. CrossEntropyLoss 내부에서 Softmax 처리 후, Loss 계산 
            loss = cross_entropy_loss(outputs, targets)

            test_loss += loss.item()

            pred_label = torch.argmax(outputs, dim = 1) # 예측 Label
            mask = (pred_label == targets).float()
            test_acc += 100 * torch.sum(mask).item() / targets.size(0)

             # 시각화용 이미지 저장 (최대 num_visualize개)
            if visualize and len(visual_imgs) < num_visualize:
                remain = num_visualize - len(visual_imgs)
                visual_imgs.extend(img[:remain])
                visual_preds.extend(pred_label[:remain])
                visual_targets.extend(targets[:remain])

        # 전체 Batch에 대한 평균값을 계산
        test_loss /= (i + 1)
        test_acc /= (i + 1)

    # 시각화 수행
    if visualize:
        visualize_predictions(visual_imgs, visual_preds, visual_targets)

    
    return test_loss, test_acc




def visualize_predictions(imgs, preds, targets):
    """
    imgs: list of Tensor (C, H, W)
    preds: list of Tensor scalar
    targets: list of Tensor scalar
    """
    class_names = ['Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise', 'Neutral']  # 예시

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for i in range(n):
        img = TF.to_pil_image(imgs[i].cpu())
        pred = class_names[preds[i].item()]
        gt = class_names[targets[i].item()]

        axes[i].imshow(img)
        axes[i].set_title(f"GT: {gt}\nPred: {pred}", fontsize=12)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()