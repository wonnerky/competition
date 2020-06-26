import torch
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(4, 10)
        self.bn1 = nn.BatchNorm1d(1)
        self.lstm2 = nn.LSTM(10,20)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(20, 1)

    def forward(self, inputs):
        out, _ = self.lstm1(inputs)
        out= self.bn1(out)
        out, _ = self.lstm2(out)
        out= self.bn2(out)
        out= self.fc1(out)

        #out=out[-1]
        return out
