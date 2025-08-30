import os
import sys
import torch
import random
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# 절대 경로 설정 -> TEAM_PROJECT 폴더로 이동
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_emo.utils.label_matching import fix_label
from shared.path import DATA_ROOT

def make_text_dataset(image_label, phase):
    """
    image_label: tensor of shape (batch_size,) containing emotion class indices (0~6)
    phase: 'train', 'val', or 'test'에 따라 다른 엑셀 파일에서 텍스트 추출
    
    반환: text_data (list of sentences) - 배치 사이즈에 맞춰 라벨에 맞는 문장 리스트에서 랜덤 선택 후 리스트 반환
    """
    # 1. 감정 인덱스와 영어 감정명 매핑
    label_to_emotion = {
        0: 'Joy',
        1: 'Sadness',
        2: 'Surprise',
        3: 'Anger',
        4: 'Fear',
        5: 'Disgust',
        6: 'Neutral'
    }
    
    # 2. phase에 따른 파일 경로 지정
    if phase == 'test':
        text_file = os.path.join(DATA_ROOT, 'text_data', 'test_text.xlsx')
    else:
        raise ValueError("phase should be 'test' for testing")
    
    # 3. 엑셀 파일에서 텍스트 데이터 불러오기
    df = pd.read_excel(text_file)
    
    # 4. 감정별 문장들을 딕셔너리로 묶기
    emotion_to_sentences = {}
    for emotion_name in label_to_emotion.values():
        emotion_to_sentences[emotion_name] = df[df['Emotion'] == emotion_name]['Sentence'].tolist()
    
    # 5. image_label에 따른 문장 무작위 선택
    text_data = []
    for label in image_label.cpu().numpy():
        emotion_name = label_to_emotion[label]
        sentences = emotion_to_sentences.get(emotion_name, [""])  # 없으면 빈문장 대체
        if len(sentences) == 0:
            text_data.append("")  # 문장이 없는 경우 빈문장
        else:
            text_data.append(random.choice(sentences))
    
    return text_data

def test(img_test_loader, mlp, img_model, text_model, num_classes=7):
    """
    Test function for the combined model
    
    Args:
        img_test_loader: Test data loader
        mlp: Pre-trained fusion model (FusionMLP)
        img_model: Pre-trained image model
        text_model: Pre-trained text model
        num_classes: Number of emotion classes
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 토크나이저 초기화
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # 모델들을 평가 모드로 설정
    img_model.eval()
    text_model.eval()
    mlp.eval()
    
    print("Using fusion model for testing")
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, data in enumerate(img_test_loader):
            # Image data
            img_data = data['image'].to(device)
            img_labels = data['emotion_label_idx'].to(device)
            targets = fix_label(img_labels).to(device)
            
            # Text data
            text_data = make_text_dataset(targets, 'test')
            
            # 텍스트 데이터 토크나이징
            encoding = tokenizer(
                text_data,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # 모델 예측
            img_output = img_model(img_data)
            text_output = text_model(input_ids, attention_mask)
            
            # Fusion model 사용
            outputs = mlp.forward(img_output, text_output)
            pred_cls = torch.argmax(outputs, dim=1)
            
            # 예측 결과 계산
            mask = (pred_cls == targets).float()
            total_correct += torch.sum(mask).item()
            total_samples += targets.size(0)
    
    accuracy = (total_correct / total_samples) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy