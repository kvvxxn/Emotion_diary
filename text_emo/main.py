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
from sklearn.model_selection import train_test_split

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

    # 하이퍼파라미터 후보 리스트

    best_val_loss = float('inf')
    best_config = None
    best_model_path = None

    # 1. 데이터 로드 & 분할 (epoch마다 중복하지 않게 여기서 한 번만 수행)
    df = pd.read_excel(file_path)
    texts = df['Sentence'].tolist()
    labels = df['Emotion'].tolist()
    emotion_to_idx = {emo: i for i, emo in enumerate(emotion_list)}
    label_vectors = [label_to_vec(l, emotion_to_idx) for l in labels]
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_vectors, test_size=0.1, random_state=42
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 2. 하이퍼파라미터 Sweep
    for lr in LEARNING_RATE:
        for wd in WEIGHT_DECAY:
            print(f"\n[Train] lr={lr} / weight_decay={wd}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EmotionClassifier(num_labels=len(emotion_list)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = torch.nn.BCEWithLogitsLoss()

            # Early Stopping 변수
            patience = 3
            best_this_val_loss = float('inf')
            es_counter = 0
            
            for epoch in range(EPOCHS):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss = evaluate(model, val_loader, criterion, device)
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # Early Stopping: val_loss 개선시만 저장, 아니면 patience 후 break
                if val_loss < best_this_val_loss:
                    best_this_val_loss = val_loss
                    es_counter = 0
                    model_path = MODEL_SAVE_PATH + f"best_tmp_lr{lr}_wd{wd}_E{EPOCHS}.pt"
                    torch.save(model.state_dict(), model_path)
                else:
                    es_counter += 1
                    if es_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # best 전체 val_loss 보다 개선(작으면) best로 갱신
            if best_this_val_loss < best_val_loss:
                best_val_loss = best_this_val_loss
                best_config = (lr, wd)
                best_model_path = model_path

    print(f"\n[Best] lr={best_config[0]}, weight_decay={best_config[1]}, val_loss={best_val_loss:.4f}")
    print(f"최적 성능 모델 경로: {best_model_path}")

    # 3. 최적 모델 다시 불러서 최종 test 평가
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(num_labels=len(emotion_list)).to(device)
    model.load_state_dict(torch.load(best_model_path))
    test_acc = test(model, val_loader, device)
    print(f"Best Val/Test Accuracy: {test_acc:.4f}")

    # 필요하면 최종 model_path를 config에서 사용하던 공식 경로로 복사/저장
    # import shutil; shutil.copy(best_model_path, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()