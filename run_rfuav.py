### Script version of RF UAV to run overnight
import os
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from helper_functions import *
from loading_functions import *

import time

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torchvision.models as models
from models import *
# from torchmetrics import F1Score
from nn_functions import runkfoldcv

## Import data -  Drone RF
main_folder = '/home/kzhou/Data/DroneRF/'
t_seg = 0.25 #ms
Xs_arr, ys_arr, y4s_arr, _ = load_dronerf_raw(main_folder, t_seg) # this loading function loads data in 2x10000 form

## Apply normalization
L_max = np.max(Xs_arr[:,1,:])
L_min = np.min(Xs_arr[:,1,:])
H_max = np.max(Xs_arr[:,0,:])
H_min = np.min(Xs_arr[:,0,:])
Maxes = np.vstack((H_max, L_max))
Mins = np.vstack((H_min, L_min))

Xs_norm = np.zeros(Xs_arr.shape)
# for ihl in range(2):
#     for i_sampe in range(Xs_norm.shape[0]):
# APPLYING GLOBAL MIN/MAX
#         Xs_norm[i_sampe,ihl,:] = (Xs_arr[i_sampe,ihl,:]-Mins[ihl])/(Maxes[ihl]-Mins[ihl])

# TO DO: COMPARE THE MAX AND MINS OF EACH SEGMENT VS OVER ALL
# reimplementing what the function in loader is doing
for i in range(Xs_arr.shape[0]): # for each segment
    for ihl in range(2):
        r = np.max(Xs_arr[i,ihl,:]) - np.min(Xs_arr[i,ihl,:]) # compute min max per sample
        m = np.min(Xs_arr[i,ihl,:])
#         print('min is', m)
#         print('range is', r)
        Xs_norm[i,ihl,:] = (Xs_arr[i,ihl,:]-m)/r

dataset = DroneData(Xs_norm, y4s_arr)

len(dataset)

# Network Hyperparameters
num_classes = len(set(dataset.yarr))
print('num classes:', num_classes)
batch_size = 128 # 128
learning_rate = 0.01
num_epochs = 10 # 0
momentum = 0.95
l2reg = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = RFUAVNet(num_classes)
model = model.to(device)

k_folds = 5
model, avg_acc, mean_f1s, mean_runtime = runkfoldcv(
    model, dataset, device, k_folds, batch_size, learning_rate, num_epochs, momentum, l2reg)
