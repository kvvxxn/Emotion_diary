import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer

from shared.path import BEST_MODEL_ROOT 
from shared.models.vit_model import vit_model
from shared.models.fusion_head import FusionMLP
from shared.models.text_emotion_classifier import EmotionClassifier

# 이미지 모델 Load
vit = vit_model()
vit.load_state_dict(torch.load(os.path.join(BEST_MODEL_ROOT, 'image_best_model.pth')))

# 텍스트 모델 Load
text_model = EmotionClassifier(num_labels=7)
text_model.load_state_dict(torch.load(os.path.join(BEST_MODEL_ROOT, 'text_best_model.pth')))
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Fusion Head Load
FusionHead = FusionMLP(hidden_dim=14, num_classes=7, dropout=0.5, batchnorm=False)

# Input Setting
np.random.seed(42)  
input_image = np.random.rand(224, 224, 3).astype(np.float32)

input_text = "오늘은 기분이 정말 좋아요!"
encoding = tokenizer(
    input_text,
    truncation=True,
    padding='max_length',
    max_length=128,
    return_tensors='pt'
)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']


# output
with torch.no_grad():
    # 둘 다 Logit 형태로 반환
    image_output = vit(input_image)
    text_output = text_model(input_ids, attention_mask)
    
    output = FusionHead(image_output, text_output)
    print(output)
