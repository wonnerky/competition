import torch
import torch.nn as nn
import dacon01.preprocessing3 as prepro

train_features, test_features, train_x, train_y, train_m, train_v = prepro.preprocessing_basic()

