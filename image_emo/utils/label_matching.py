import os
import json
import torch

raw_label = ['amusement', 'awe', 'contentment', 'excitement', 
            'anger', 'disgust', 'fear', 'sadness']

# Cross-Entropy loss를 사용하려면 class index를 가져야 함
"""
0 ~ 6까지의 label은 아래와 같다.
    - Joy, Sadness, Suprise, Anger, Fear, Disgust, Neutral

EmoSet의 원래 8개의 감정 Label을 아래와 같이 7개의 감정 Label로 변환하였다.
    - amusement -> Joy
    - awe -> Surprise
    - contentment -> Neutral
    - excitement -> Joy
    - anger -> Anger
    - disgust -> Disgust
    - fear -> Fear
    - sadness -> Sadness  
"""
scores = torch.tensor([0, 2, 6, 0, 3, 5, 4, 1])

label_to_emotion = {
    0: 'Joy',
    1: 'Sadness',
    2: 'Surprise',
    3: 'Anger',
    4: 'Fear',
    5: 'Disgust',
    6: 'Sadness'
}

    
def label_to_score(v):
    # Label Tensor를 입력받아 7개 class에 대한 점수 형식으로 반환
    answer = scores[v]
    return answer

def answer_to_json(score):
    # 위의 매칭 결과를 JSON 파일로 저장
    # 라벨 매칭 (8 Class -> 7 Class)
    label_dict = {}
    label_dict = dict(zip(raw_label, score))

    with open(os.path.join(os.path.dirname(__file__), 'answer.json'), 'w') as f:
        json.dump(label_dict, f, indent = 4)

# Score vector 조정 -> 직접 실행했을 때만 가능
if __name__ == "__main__":
    # Tensor가 JSON 파일로의 쓰기 지원이 안 됨
    score = scores.tolist()
    answer_to_json()
        
        

