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
from preprocessing3 import preprocessing_basic, DaconDataset, get_min_max, norm_to_val


path = '../data/KAERI_dataset/'
train_feature, test_feature, x, y, m, v = preprocessing_basic(path, isLabel=False)
test_feature = test_feature.reshape(-1)
for i in range(len(test_feature)):
    test_feature[i] = float(test_feature[i])
test_feature = test_feature.reshape(-1, 1, 4)
test_dataset = DaconDataset(test_feature)
batch_size = 375
test_loader = data.DataLoader(test_dataset, batch_size=batch_size,)
model_path = './ckpt/0706/2/49.pth'
model = WhModel()
model.load_state_dict(torch.load(model_path))
model.eval()
print(x)
exit()
for batch_idx, sample in enumerate(test_loader):
    features = sample["input"]
    max_x, min_x = get_min_max(x)
    features = Variable(features.float())
    output_logits = model(features)
    output_logits = output_logits[-1]
    output = output_logits.tolist()[0][0]
    print(output)
    result = norm_to_val(output, max_x, min_x)
    print(f'{batch_idx}: {result}')
