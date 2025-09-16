import os
import time
import torch
import numpy as np
from transformers import BertTokenizer

from shared.path import BEST_MODEL_ROOT 
from shared.models.vit_model import vit_model
from shared.models.fusion_head import FusionMLP
from shared.models.text_emotion_classifier import EmotionClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 모델 Load
vit, _ = vit_model()
vit.to(device).eval()
vit.load_state_dict(torch.load(
    os.path.join(BEST_MODEL_ROOT, 'image_best_model.pth'),
    map_location=device
)) 

# 텍스트 모델 Load
text_model = EmotionClassifier(num_labels=7)
text_model.load_state_dict(torch.load(
    os.path.join(BEST_MODEL_ROOT, 'text_best_model.pt'),
    map_location=device
))
text_model.to(device).eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Fusion Head Load
FusionHead = FusionMLP(hidden_dim=14, num_classes=7, dropout=0.5, batchnorm=False)
FusionHead.to(device).eval()

# Input Setting
np.random.seed(42)  
input_image = np.random.rand(224, 224, 3).astype(np.float32)
input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).to(device)

input_text = "오늘은 기분이 정말 좋아요!"
encoding = tokenizer(
    input_text,
    truncation=True,
    padding='max_length',
    max_length=128,
    return_tensors='pt'
)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)


# ----------------------
# 2) 시간 측정 유틸
# ----------------------
def _sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

# ----------------------
# 3) 추론 + 단계별/총 시간
# ----------------------
_sync()
t0_total = time.perf_counter()

# (a) 이미지 분기
t0_img = time.perf_counter()
with torch.no_grad():
    # 둘 다 Logit 형태로 반환
    image_output = vit(input_image)      # 기대 shape: [1, 7]
_sync()
t_img = time.perf_counter() - t0_img

# (b) 텍스트 분기
t0_txt = time.perf_counter()
with torch.no_grad():
    # 둘 다 Logit 형태로 반환
    text_output = text_model(input_ids, attention_mask)  # 기대 shape: [1, 7]
_sync()
t_txt = time.perf_counter() - t0_txt

# (c) 퓨전
t0_fuse = time.perf_counter()
with torch.no_grad():
    # 둘 다 Logit 형태로 반환
    output = FusionHead(image_output, text_output)    # shape: [1, 7] (로짓)
_sync()
t_fuse = time.perf_counter() - t0_fuse

t_total = time.perf_counter() - t0_total

# ----------------------
# 4) 결과 출력
# ----------------------
print("output logits:", output)   # 로짓 (softmax 미적용)
print(f"image_path: {t_img*1000:.2f} ms")
print(f"text_path:  {t_txt*1000:.2f} ms")
print(f"fusion:     {t_fuse*1000:.2f} ms")
print(f"TOTAL:      {t_total*1000:.2f} ms")

