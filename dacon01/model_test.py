import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(4, 1)
        self.fc1 = nn.Linear(4, 1)

    def forward(self, inputs):
        out, _ = self.lstm1(inputs)
        return out
