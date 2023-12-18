
import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *


class cnn3d(nn.Module):
    def __init__(self, num_classes, in_channels):

        
        super(cnn3d, self).__init__()

        self.conv_layer1 = self._make_conv_layer(in_channels, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)
        self.conv_layer5=nn.Conv3d(256, 256, kernel_size=(3, 4, 3), padding=0)
        
        self.fc5 = nn.Linear(256, 256)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(p=0.15)        
        self.fc6 = nn.Linear(256, 124)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(124)
        
        self.drop=nn.Dropout(p=0.15)
        self.fc7 = nn.Linear(124, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x1=x
        x = self.fc7(x)

        return x

# class cnn3d(nn.Module):

#     def __init__(self):
#         super(cnn3d, self).__init__()
#         self.conv1 = self._conv_layer_set(1, 16)
#         self.conv2 = self._conv_layer_set(16, 32)
#         self.conv3 = self._conv_layer_set(32, 64)
#         self.fc1 = nn.Linear(10*10*10*64, 128)
#         self.fc2 = nn.Linear(128, 20)
#         self.relu = nn.LeakyReLU()
#         self.conv1_bn = nn.BatchNorm3d(16)
#         self.conv2_bn = nn.BatchNorm3d(32)
#         self.conv3_bn = nn.BatchNorm3d(64)
#         self.fc1_bn = nn.BatchNorm1d(128)
#         self.drop = nn.Dropout(p=0.3)

#     def _conv_layer_set(self, in_channels, out_channels):
#         conv_layer = nn.Sequential(
#             nn.Conv3d(
#                 in_channels, 
#                 out_channels, 
#                 kernel_size=(3, 3, 3), 
#                 stride=1,
#                 padding=0,
#                 ),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
#             )
#         return conv_layer

#     def forward(self, x):
#         #print('input shape:', x.shape)
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = self.conv2(x)
#         x = self.conv2_bn(x)
#         x = self.conv3(x)
#         x = self.conv3_bn(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc1_bn(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         #print('output shape:', x.shape)

#         return x


## CNN with two conv layers, global average pooling, and two dense layers.
class Net(nn.Module):

    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.glob_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(torch.nn.Module): 
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(1749600, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


