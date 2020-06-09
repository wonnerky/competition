import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dacon01.preprocessing import preprocessing_concat_seq, preprocessing_basic
from sklearn.preprocessing import MultiLabelBinarizer

torch.manual_seed(1)

train_features, test_features, train_x, train_y, train_m, train_v = preprocessing_concat_seq()


x_train = train_features # [[x,y,z],[x,y,z]] 각 프레임들

y_train = train_x #300개 있어야함
print(y_train)
#x_train=list(map(float,x_train[]))

'''
24 : 오른손, 22 : 왼손, 20 : 오른발, 16 : 왼발
'''
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
#y_train=y_train.unsqueeze(1)


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1500, 20)
        self.linear2 = nn.Linear(20, 15)
        self.linear3 = nn.Linear(15, 9)
        # self.linear4 = nn.Linear(600, 300)
        # self.linear5 = nn.Linear(300,9) # 관절개수*클래스개수

        self.relu=nn.ReLU()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):

        x=self.linear1(x)
        x=self.relu(x)
        x=self.drop(x)

        x=self.linear2(x)
        x=self.relu(x)
        x=self.drop(x)

        x=self.linear3(x)
        x=self.relu(x)
        x=self.drop(x)

        # x=self.linear4(x)
        # x=self.relu(x)
        # x=self.drop(x)
        #
        # x=self.linear5(x)
        # x=self.relu(x)
        # x=self.drop(x)


        return x

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.Adam(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))