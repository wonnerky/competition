import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from model_test import TestModel

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
    train_features = train_features.reshape(-1, 375, 4)
    test_features = test_features.reshape(-1, 375, 4)
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


if __name__ == '__main__':
    train_feature, test_feature, x, y, m, v = preprocessing_basic()

    model = TestModel()
    model.eval()
    print(train_feature[0].shape)
    input = torch.tensor(train_feature[0])
    input = torch.unsqueeze(input, 0)
    print(input.shape)
    print(input.size(0))
    out = model(train_feature[0])
    print(out)