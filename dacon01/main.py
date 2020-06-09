import numpy as np
from preprocessing import preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


test_features, train_features, train_x_label, train_y_label, train_m_label, train_v_label = preprocessing()
print(train_features.shape)
train_features = train_features.reshape(-1,1500)
print(train_features.shape)
train_x = train_features
train_x = torch.FloatTensor(train_x)
print(train_x)

label_x = train_x_label
label_x = torch.FloatTensor(label_x)
print(label_x)
#
# W = torch.zeros((1500, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# torch.manual_seed(1)
#
# model = nn.Linear(1500,1)
#
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# nb_epochs = 2000
# for epoch in range(nb_epochs+1):
#
#     # H(x) 계산
#     prediction = model(train_x)
#     # model(x_train)은 model.forward(x_train)와 동일함.
#
#     # cost 계산
#     cost = F.mse_loss(prediction, label_x) # <== 파이토치에서 제공하는 평균 제곱 오차 함수
#
#     # cost로 H(x) 개선하는 부분
#     # gradient를 0으로 초기화
#     optimizer.zero_grad()
#     # 비용 함수를 미분하여 gradient 계산
#     cost.backward()
#     # W와 b를 업데이트
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         # 100번마다 로그 출력
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))
#
#


