import torch
import torch.nn as nn
import dacon01.preprocessing as prepro

train_features, test_features, train_x, train_y, train_m, train_v = prepro.preprocessing_basic()

print(train_features[:20])
print(train_features.size)

