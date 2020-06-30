import torch
import torch.nn as nn
import dacon01.preprocessing3 as prepro

train_features, test_features, train_x, train_y, train_m, train_v = prepro.preprocessing_basic()

print(train_features[:20])
print(train_features.size)
print(set(train_x))
print(set(train_y))
print(set(train_m))
print(set(train_v))
