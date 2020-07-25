import pandas as pd
import numpy as np
import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils import data
import matplotlib.pyplot as plt
from model_test3 import TestModel
import torchnet as tnt
from tqdm import tqdm
from torch.autograd import Variable
import pickle


def preprocessing_basic(file_path, isFeatrue=True, isLabel=True):
    path = file_path
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
    if isFeatrue:
        file_name = "norm_train_features.txt"
        if os.path.isfile(file_path + 'norm_train_features.txt'):
            train_features = pickle_load(file_path, file_name)
            print("train feature data load complete")
        else:
            print("train feature norm processing")
            train_features = norm_feature(train_features)
            pickle_save(file_path, file_name, train_features)
            print("save norm train feature complete")
        file_name = "norm_test_features.txt"
        if os.path.isfile(f'{file_path}{file_name}'):
            test_features = pickle_load(file_path, file_name)
            print("test feature data load complete")
        else:
            print("test feature norm processing")
            test_features = norm_feature(test_features)
            pickle_save(file_path, file_name, test_features)
            print("save norm test feature complete")

    train_x = train_target[:,1]
    train_y = train_target[:,2]
    train_m = train_target[:,3]
    train_v = train_target[:,-1]
    if isLabel:
        file_name = "norm_label_x.txt"
        if os.path.isfile(f'{file_path}{file_name}'):
            train_x = pickle_load(file_path, file_name)
            print("label x data load complete")
        else:
            print("label x norm processing")
            train_x = norm_label(train_x)
            pickle_save(file_path, file_name, train_x)
            print("save norm label x complete")
        file_name = "norm_label_y.txt"
        if os.path.isfile(f'{file_path}{file_name}'):
            train_y = pickle_load(file_path, file_name)
            print("label y data load complete")
        else:
            print("label y norm processing")
            train_y = norm_label(train_y)
            pickle_save(file_path, file_name, train_y)
            print("save norm label y complete")
        file_name = "norm_label_m.txt"
        if os.path.isfile(f'{file_path}{file_name}'):
            train_m = pickle_load(file_path, file_name)
            print("label m data load complete")
        else:
            print("label m norm processing")
            train_m = norm_label(train_m)
            pickle_save(file_path, file_name, train_m)
            print("save norm label m complete")
        file_name = "norm_label_v.txt"
        if os.path.isfile(f'{file_path}{file_name}'):
            train_v = pickle_load(file_path, file_name)
            print("label v data load complete")
        else:
            print("label v norm processing")
            train_v = norm_label(train_v)
            pickle_save(file_path, file_name, train_v)
            print("save norm label v complete")

    return train_features, test_features, train_x, train_y, train_m, train_v


def pickle_load(file_path, file_name):
    with open(f'{file_path}{file_name}', 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_save(file_path, file_name, data):
    with open(f'{file_path}{file_name}', 'wb') as f:
        pickle.dump(data, f)


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

def _norm_feature(feature):
    norm_feature = []
    for idx, x in enumerate(feature):
        norm_feature.append((x - np.mean(feature)) / np.var(feature))
    return norm_feature

def get_min_max(labels):
    return np.max(labels), np.min(labels)


def norm_to_val(output, max, min):
    result = (output * (max - min)) + min
    return result

def norm_feature(features):
    norm_feature = []
    features = features.reshape(-1, 1500)
    for i in tqdm(range(len(features))):
        norm_feature.append(_norm_feature(features[i]))
    norm_feature = np.array(norm_feature)
    return norm_feature.reshape(-1, 1, 4)

def norm_label(labels):
    norm_labels = []
    for idx, x in enumerate(tqdm(labels)):
        norm_labels.append((x - np.min(labels)) / (np.max(labels) - np.min(labels)))
    return np.array(norm_labels)

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
    for i in range(len(x)):
        x[i] = float(x[i])
    train_feature = train_feature.reshape(-1)
    for i in range(len(train_feature)):
        train_feature[i] = float(train_feature[i])
    train_feature = train_feature.reshape(-1,1,4)
    train_dataset_x = DaconDataset(train_feature, labels=x)
    batch_size = 375
    train_loader_x = data.DataLoader(train_dataset_x, batch_size=batch_size,)
    model = TestModel()
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

            print(f"Epoch: {e}  , Loss: {meter_loss.value():.4f}")   #  , Accuracy: {acc.value()[0]:.2f}")

        filename = 'finalized_model'+str(e)+'.sav'
        pickle.dump(model, open(filename, 'wb'))

        torch.save(model.state_dict(),
                   'C:\\Users\\SYM\\PycharmProjects\\competition\\dacon01\\ckpt\\finalized_model2'+str(e)+'.pth')




