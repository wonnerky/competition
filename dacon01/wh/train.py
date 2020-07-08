import torch, os, sys
import torch.nn as nn
import numpy as np
import csv, pickle
import pandas as pd
from torch.autograd import Variable
import torchnet as tnt
import torch.optim as optim
from torch.utils import data
from model_wh import WhModel

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model_test3 import TestModel
from preprocessing3 import preprocessing_basic, DaconDataset


path = '../data/KAERI_dataset/'
train_feature, test_feature, x, y, m, v = preprocessing_basic(path)
for i in range(len(x)):
    x[i] = float(x[i])
train_feature = train_feature.reshape(-1)
for i in range(len(train_feature)):
    train_feature[i] = float(train_feature[i])
train_feature = train_feature.reshape(-1, 1, 4)
train_dataset_x = DaconDataset(train_feature, labels=x)
batch_size = 375
train_loader_x = data.DataLoader(train_dataset_x, batch_size=batch_size, )
model = WhModel()
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.MSELoss()
# model.cuda()
epoch = 50

for e in range(epoch):

    # acc = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_loss = tnt.meter.MSEMeter()

    for batch_idx, sample in enumerate(train_loader_x):
        features = sample["input"]
        label = sample["label"]

        # input_features_var = Variable(features.cuda())
        # input_label_var = Variable(label[-1].cuda())
        label = label[-1]
        label = label.unsqueeze(0).unsqueeze(0)

        features = Variable(features.float())
        label = Variable(label.float())

        # output_logits = model(input_features_var)
        output_logits = model(features)
        output_logits = output_logits[-1]
        # loss = criterion(output_logits, input_label_var)
        loss = criterion(output_logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter_loss.add(loss.data, label.data)
        # acc.add(output_logits.data, label.data)

        if batch_idx % 100 == 0:
            print(f"Epoch: {e} ,  Batch: {batch_idx} ,  Loss: {meter_loss.value():.4f}")  # , Accuracy: {acc.value()[0]:.2f}")

    filename = 'finalized_model' + str(e) + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    save_path = './ckpt/0706/'
    torch.save(model.state_dict(), save_path + str(e) + '.pth')