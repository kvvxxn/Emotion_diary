import torch
import torch.nn as nn
import torch.nn.functional as F

# 두 모델에서의 벡터 합치기 위한 2 Layer MLP 정의
# BN -> FC -> BN -> ReLU -> FC
class FusionMLP(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.5, batchnorm: bool = False):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(14) if batchnorm else nn.Identity()
        self.fc1 = torch.nn.Linear(14, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    
    def forward(self, img_vec: torch.Tensor, text_vec: torch.Tensor):
        # 두 벡터 연결
        input = torch.concatenate([img_vec, text_vec], dim=1)

        input = self.bn1(input)
        
        hidden = self.fc1(input)
        hidden = self.bn2(hidden)
        hidden = F.relu(hidden, inplace=True)
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)

        return output