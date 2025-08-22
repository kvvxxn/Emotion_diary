import os
import sys
import torch
from transformers import BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.datasets.text_emoset import EmotionDataset
from shared.models.emotion_classifier import EmotionClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # combine 상위
SHARED_ROOT = os.path.join(PROJECT_ROOT, 'shared')
TEXT_BEST_MODEL_PATH = os.path.join(SHARED_ROOT, 'best_model', 'text_best_model.pt')

# 모델 준비
model = EmotionClassifier(num_labels=7)
model.load_state_dict(torch.load(TEXT_BEST_MODEL_PATH), map_location=torch.device('cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 토크나이저
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

text = "오늘은 기분이 정말 좋아요!"
encoding = tokenizer(
    text,
    truncation=True,
    padding='max_length',
    max_length=128,
    return_tensors='pt'
)

# 추론
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

with torch.no_grad():
    logits = model(input_ids, attention_mask)
    probs = torch.sigmoid(logits)
    print("모델 예측:", probs)
