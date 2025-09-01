import numpy as np
from color_matching import EMOTION_COLOR_MAP, EMOTION_LABELS

def recommend_emotion_colors(emotion_vector, method='mix'):
    """
    emotion_vector: [float] shape=(7,) 모델의 7D 감정 벡터 (softmax or 확률합 1)
    method: 'top1' or 'mix'
    return: 추천 HEX 색상 or 색상 리스트
    """
    emotion_vector = np.array(emotion_vector)
    if method == 'top1':
        idx = int(np.argmax(emotion_vector))
        return {
            "color": EMOTION_COLOR_MAP[idx],
            "label": EMOTION_LABELS[idx],
            "score": float(emotion_vector[idx])
        }
    elif method == 'mix':
        indices = emotion_vector.argsort()[-3:][::-1]
        return [
            {
                "color": EMOTION_COLOR_MAP[i],
                "label": EMOTION_LABELS[i],
                "score": float(emotion_vector[i])
            }
            for i in indices
        ]
    else:
        raise ValueError("method must be 'top1' or 'mix'")