import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# import the torch packages
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

import torchvision.models as models

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# import custom functions
from helper_functions import *
from latency_helpers import *
from loading_functions import *

from Torch_Models import *

from file_paths import *

from nn_functions import runkfoldcv
# from torchsummary import summary

feat_name = 'SPEC' # 'RAW' or 'PSD' or 'SPEC'
t_seg = 20
n_per_seg = 1024

output_name = 'drones'
feat_format = 'IMG'# ARR, IMG
which_dataset = 'dronerf'
output_tensor = True

# dataset specific parameters
drrf_highlow = 'L'
drde_ints = ['WIFI','CLEAN','BLUE','BOTH']

if which_dataset == 'dronerf':
    print('Loading DroneRF Dataset')
    dataset = DroneRFTorch(dronerf_feat_path, feat_name, t_seg, n_per_seg,
                       feat_format, output_name, output_tensor, drrf_highlow)
elif which_dataset == 'dronedetect':
    print('Loading DroneDetect Dataset')
    dataset = DroneDetectTorch(dronedetect_feat_path, feat_name, t_seg, n_per_seg, feat_format,
                                    output_name, output_tensor, drde_ints)
        
num_classes = len(dataset.unique_labels)
which_model = 'vgg' # or 'resnet'
if which_model == 'vgg':
    Model = VGGFC(num_classes)
elif which_model == 'resnet':
    Model = ResNetFC(num_classes)
elif which_model == '1dconv':
    Model = RFUAVNet(num_classes)

test_samp = dataset.__getitem__(40)[0]
test_samp = torch.unsqueeze(test_samp, 0)
Model(test_samp)


k_folds = 5

batch_size = 128 # 128
learning_rate = 0.01
num_epochs = 10 # 0
momentum = 0.95
l2reg = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trainedModel, res_acc, res_f1, res_runtime = runkfoldcv(Model, dataset, device, k_folds, batch_size, learning_rate, num_epochs, momentum, l2reg)