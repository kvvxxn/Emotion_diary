import os
import sys
import torch

# 절대 경로 설정 -> TEAM_PROJECT 폴더로 이동
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_emo.models.vit_model import vit_model

def two_models_combining():
    # TODO: 이미지, 텍스트 입력 받기
    input_image = input()

    image_model_path = os.path.join('image_emo', 'image_best_model.pth')

    # TODO: Text Model 불러오기 / 변수명: text_model
    
    # Image Model 불러오기
    vit, _ = vit_model()
    vit.load_dict(torch.load(image_model_path))

    vit.eval()

    with torch.no_grad():
        image_vec = vit.forward_features(input_image)

    # TODO: Text Model 돌리고 text_vec 얻기

    # 두 벡터 사이의 코사인 유사도
    emo_vec = torch.cosine_similarity(image_vec, text_vec)

    return emo_vec
