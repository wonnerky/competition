import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=5)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=5)
        self.lstm3 = nn.LSTM(input_size=1, hidden_size=5)
        self.lstm4 = nn.LSTM(input_size=1, hidden_size=5)
        self.fc1 = nn.Linear(4, 1)

    def forward(self, inputs, ):
        out1 = self.lstm1(inputs[0])
        out2 = self.lstm2(inputs[1])
        out3 = self.lstm2(inputs[2])
        out4 = self.lstm2(inputs[3])
        out = self.fc1(torch.cat((out1,out2,out3,out4), dim=1))
        return out


