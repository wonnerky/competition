import torch
import torch.nn as nn


class WhModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 5, 2)
        self.lstm2 = nn.LSTM(1, 5, 2)
        self.lstm3 = nn.LSTM(1, 5, 2)
        self.lstm4 = nn.LSTM(1, 5, 2)
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, inputs):
        out1, _ = self.lstm1(torch.unsqueeze(inputs[:,:,0], 1))
        out2, _ = self.lstm1(torch.unsqueeze(inputs[:,:,1], 1))
        out3, _ = self.lstm1(torch.unsqueeze(inputs[:,:,2], 1))
        out4, _ = self.lstm1(torch.unsqueeze(inputs[:,:,3], 1))
        # concat
        out_cat = torch.cat((out1, out2, out3, out4), 2)
        out = self.fc1(out_cat)
        out = self.fc2(out)


        #out=out[-1]
        return out
