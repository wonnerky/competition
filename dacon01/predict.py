import os
import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as DD
import torchnet as tnt
from preprocessing import preprocessing_basic, DaconDataset, get_min_max, norm_to_val
from model_test3 import TestModel
import pickle
from torch.utils import data

model=TestModel()
model.load_state_dict(torch.load('C:\\Users\\SYM\\PycharmProjects\\competition\\dacon01\\ckpt\\finalized_model219.pth'))
model.eval()


train_features, test_features, train_x, train_y, train_m, train_v = preprocessing_basic(norm_train=False, norm_test=True, norm_labels=False)

test_features = test_features.reshape(-1)
for i in range(len(test_features)):
    test_features[i] = float(test_features[i])
test_features = test_features.reshape(-1,1,4)
test_dataset_x = DaconDataset(test_features)
batch_size = 375
test_loader_x = data.DataLoader(test_dataset_x, batch_size=batch_size,)
x_max, x_min = get_min_max(train_x)
for batch_idx, sample in enumerate(test_loader_x):
    input = sample['input']

    output_logits = model(input.float())

    output_logits = output_logits[-1]
    print(output_logits)
    #print(norm_to_val(output_logits, x_max, x_min))

