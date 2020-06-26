import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils import data
import matplotlib.pyplot as plt
from model_test import TestModel
import torchnet as tnt
from tqdm import tqdm
from torch.autograd import Variable



def preprocessing_basic():
    path = './data/KAERI_dataset/'
    test_features = pd.read_csv(path + 'test_features.csv')
    train_features = pd.read_csv(path + 'train_features.csv')
    train_target = pd.read_csv(path + 'train_target.csv')

    test_features = test_features.values
    train_features = train_features.values
    train_target = train_target.values

    train_features = train_features[:,2:]
    test_features = test_features[:,2:]
    train_features = train_features.reshape(-1, 1, 4)
    test_features = test_features.reshape(-1, 1, 4)
    train_features = norm(train_features)
    test_features = norm(test_features)
    train_x = train_target[:,1]
    train_y = train_target[:,2]
    train_m = train_target[:,3]
    train_v = train_target[:,-1]

    return train_features, test_features, train_x, train_y, train_m, train_v

def preprocessing_concat_seq():
    train_features, test_features, train_x, train_y, train_m, train_v = preprocessing_basic()
    # li = []
    # for i in range(375):
    #     for j in train_features[i]:
    #         li.append(j)
    train_features = train_features.reshape(-1, 1500)
    test_features = test_features.reshape(-1, 1500)
    train_x = to_ont_hot(train_x)
    train_y = to_ont_hot(train_y)
    train_m = to_ont_hot(train_m)
    train_v = to_ont_hot(train_v)

    # print(train_features.shape)
    # print(train_features)
    # print(np.array_equal(li, train_features[0]))
    return train_features, test_features, train_x, train_y, train_m, train_v

def _norm(feature):
    norm_feature = []
    for idx, x in enumerate(feature):
        norm_feature.append((x - np.mean(feature)) / np.var(feature))
    return norm_feature

def norm(features):
    norm_feature = []
    features = features.reshape(-1, 1500)
    for i in tqdm(range(len(features))):
        norm_feature.append(_norm(features[i]))
    norm_feature = np.array(norm_feature)
    return norm_feature.reshape(-1, 1, 4)


def toString(array):
    array = array.tolist()
    for i in range(len(array)):
        array[i] = str(array[i])
    return array


def to_ont_hot(array):
    array_set = list(set(array))
    for i in range(len(array)):
        array[i] = array_set.index(array[i])
    return array

# feature 는 index에 맞게 값이 나옴. 2800 x 375 x 4 . label 은 2800 x 1. 375 마다 label 값이 바뀌게 설정
class DaconDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
        self.len_features = len(features)

    def __getitem__(self, index):
        feature = self.features[index]
        if self.labels is not None:
            idx = index // 375
            label = self.labels[idx]
            sample = {'input': feature, 'label': label}
        else:
            sample = {'input': feature}
        return sample

    def __len__(self):
        return self.len_features

if __name__ == '__main__':
    train_feature, test_feature, x, y, m, v = preprocessing_basic()
    print(train_feature.shape)
    print(x.shape)
    for i in range(len(x)):
        x[i] = float(x[i])
    train_feature = train_feature.reshape(-1)
    print(train_feature.shape)
    for i in range(len(train_feature)):
        train_feature[i] = float(train_feature[i])
    train_feature = train_feature.reshape(-1,1,4)
    print(train_feature.shape)
    train_dataset_x = DaconDataset(train_feature, x)
    batch_size = 375
    train_loader_x = data.DataLoader(train_dataset_x, batch_size=batch_size,)
    model = TestModel()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    # model.cuda()
    epoch = 50

    for e in range(epoch):

        acc = tnt.meter.ClassErrorMeter(accuracy=True)
        meter_loss = tnt.meter.AverageValueMeter()

        for batch_idx, sample in enumerate(train_loader_x):
            features = sample["input"]
            label = sample["label"]
            # input_features_var = Variable(features.cuda())
            # input_label_var = Variable(label[-1].cuda())
            label = label[-1]

            features = Variable(features.float())
            label = Variable(label.float())


            # output_logits = model(input_features_var)
            output_logits = model(features)
            # loss = criterion(output_logits, input_label_var)
            loss = criterion(output_logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter_loss.add(loss.data)
            acc.add(output_logits.data, label.data)

            print(f"Epoch: {e}  , Loss: {meter_loss.value()[0]:.4f}  , Accuracy: {acc.value()[0]:.2f}")




import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

torch.save(model.state_dict(),'C:\\Users\\SYM\\PycharmProjects\\competition\\dacon01\\finalized_model2.pth')
