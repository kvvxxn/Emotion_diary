# 전체 감정 분류 모델 학습 코드

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 1. 전처리 단계
# 감정 레이블 정의
emotion_list = ['행복', '슬픔', '놀람', '분노', '공포', '혐오', '중립']
emotion_to_idx = {emo: i for i, emo in enumerate(emotion_list)}

# 엑셀 데이터 로딩
file_path = os.path.join("data", "한국어 감정 정보가 포함된 단발성 대화 데이터셋", "한국어_단발성_대화_데이터셋.xlsx")
df = pd.read_excel(file_path)

texts = df['Sentence'].tolist()
labels = df['Emotion'].tolist()

# 레이블을 7차원 벡터로 변환
def label_to_vec(label):
    vec = np.zeros(len(emotion_list), dtype=np.float32)
    if label in emotion_to_idx:
        vec[emotion_to_idx[label]] = 1.0
    return vec

label_vectors = [label_to_vec(label) for label in labels]

# 학습/검증 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, label_vectors, test_size=0.1, random_state=42
)


# 2. Dataset 클래스 정의
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Dataset 준비
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. 모델 정의
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 7)  # 7차원 출력

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.classifier(x)

# 4. 학습 함수 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# 5. 학습 루프
epochs = 3
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss = train_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), "korean_emotion_mbert.pt")
print("모델 저장 완료: korean_emotion_mbert.pt")
