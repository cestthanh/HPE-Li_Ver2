"""
regression.py
Đầu hồi quy (regression head): vector đặc trưng phẳng -> 51 giá trị -> (17 khớp × 3).
"""
import torch.nn as nn


class RegressionHead(nn.Module):
    """MLP 3 lớp: (B, input_dim) -> (B, output_dim).

    Với cấu hình mặc định:
        input_dim  = 3584  (= 256 × 14 × 1, đặc trưng sau feature extractor)
        output_dim = 51    (= 17 khớp × 3 tọa độ)
    """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)                       # (B, 3584)
        x = self.dropout(self.relu(self.fc1(x)))           # -> hidden
        x = self.dropout(self.relu(self.bn(self.fc2(x))))  # -> hidden * 2
        return self.fc3(x)                                 # -> 51
