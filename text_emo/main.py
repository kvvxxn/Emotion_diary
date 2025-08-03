import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config.config import *
from datasets.emotion_dataset import EmotionDataset
from models.emotion_classifier import EmotionClassifier
from pipeline.train import train_epoch, evaluate
from pipeline.test import test

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def label_to_vec(label, emotion_to_idx):
    vec = np.zeros(len(emotion_to_idx), dtype=np.float32)
    if label in emotion_to_idx:
        vec[emotion_to_idx[label]] = 1.0
    return vec

def main():
    set_seed(42)

    # 1. 데이터 로드
    df = pd.read_excel(file_path)
    texts = df['Sentence'].tolist()
    labels = df['Emotion'].tolist()

    emotion_to_idx = {emo: i for i, emo in enumerate(emotion_list)}
    label_vectors = [label_to_vec(l, emotion_to_idx) for l in labels]

    # 2. 학습/검증/테스트 분할
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_vectors, test_size=0.1, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # 3. Dataset & DataLoader
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 4. 모델 및 최적화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(num_labels=len(emotion_list)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 5. 학습/검증 루프
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 6. 저장
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"모델 저장 완료: {MODEL_SAVE_PATH}")

    # 7. 테스트 예시
    test_acc = test(model, val_loader, device)
    print(f"Val/Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()