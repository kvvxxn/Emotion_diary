import os
import sys
import timm
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_emo.config.config import model_num_emotion_classes

def vit_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model loading - vit_patch16_224, version: 1.0.17
    vit = timm.models.vit_base_patch16_224(pretrained=True, num_classes = model_num_emotion_classes).to(device)
    
    # Loss function
    cross_entropy_loss = nn.CrossEntropyLoss()

    # model, loss function object를 return
    return vit, cross_entropy_loss