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
import torch
import numpy as np
import torchvision.models

import random
import time

class ContrastiveEncoder(nn.Module):
    def __init__(self, num_classes, to_augment=True):
        super(ContrastiveEncoder,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=24)
#         self.lstm1 = nn.LSTM(input_size=419407, num_layers=2, hidden_size=10, dtype=torch.float64, batch_first=True)
#         self.lstm1 = nn.LSTM(419407, 128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8)
        self.mpool = nn.MaxPool2d(kernel_size=128)
        self.encode_lin1 = nn.Linear(in_features=3276, out_features=128, bias=False)
        
        self.proj_lin1 = nn.Linear(in_features=128, out_features=128, bias=False)
        self.proj_lin2 = nn.Linear(in_features=128, out_features=num_classes, bias=False)
        
        self.thetas = [0, np.pi/2, np.pi, 3*np.pi/2]
        self.n_aug = 2
        
        self.to_augment = to_augment
        
    def forward(self, x):
        if self.to_augment:
            samp = augmentData(x, self.thetas)
        else:
            samp = x
        ri = self.encode(samp)
        if self.to_augment:
            return ri
        
        zi = self.projectionHead(ri)
#         print(zi.shape)
        return zi
    
    def encode(self,x):
        x = self.conv1(x)
#         print('past conv1:', x.shape)
#         x, hidden = self.lstm1(x)
#         print('past lstm:',len(x))
        x = self.conv2(x)
#         print('past conv2:', x.shape)
        x = self.mpool(x)
        
        x = self.encode_lin1(x)
        x = torch.squeeze(x,1)
        return x
    
        
    def projectionHead(self,x):
        x = self.proj_lin1(x)
        x = nn.ReLU()(x)
        x = self.proj_lin2(x)
        return x
    
    
## USE VGG19 TRANSFER LEARNING FOR encoder on spec images
class TransferEncoder(nn.Module):
    def __init__(self, num_classes, to_augment=True):
        super(TransferEncoder,self).__init__()
        
        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        modules=list(self.vgg16.children())[:-1]
        self.vggmodel=nn.Sequential(*modules)

        for p in self.vggmodel.parameters():
            p.requires_grad = False
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=24)
#         self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8)
        self.mpool = nn.MaxPool1d(kernel_size=128)
        self.encode_lin1 = nn.Linear(in_features=195, out_features=128, bias=False)
        
        self.proj_lin1 = nn.Linear(in_features=128, out_features=128, bias=False)
        self.proj_lin2 = nn.Linear(in_features=128, out_features=num_classes, bias=False)
        
        self.to_augment = to_augment
        self.gaussian_noise = AddGaussianNoise(0,1)
        
        
    def forward(self, x):
        print('input size:', x.shape)
            
        if self.to_augment:
            samp = self.gaussian_noise(x) ### Add noise to image
        else:
            samp = x
        ri = self.encode(samp)
        print('ri shape:', ri.shape)
        if self.to_augment:
            return ri
        zi = self.projectionHead(ri)
        print('zi shape:', zi.shape)
        
        return zi
    
    def encode(self,x):
        x = self.vggmodel(x)
#         print('after vgg shape:', x.shape)
        x = x.reshape(-1,25088)
        x = x.unsqueeze(1)
#         print('after reshape shape:', x.shape)
        x = self.conv1(x)
#         print('after cov1', x.shape)
        x = self.mpool(x)
        
        x = self.encode_lin1(x)
        x = torch.squeeze(x,1)
        return x
    
        
    def projectionHead(self,x):
        x = self.proj_lin1(x)
        x = nn.ReLU()(x)
        x = self.proj_lin2(x)
        return x
    
        
def augmentData(x, thetas):
#         for k in range(n_aug):
    the = random.sample(thetas,1)
    c, s = np.cos(the), np.sin(the)
    Rotm = torch.tensor(np.float32([[c, -s], [s, c]]))
    Rotm = Rotm.reshape(2,2)
    Rotm = Rotm.cuda()
    aug_samp = Rotm.matmul(x) 
#         torch.matmul(Rotm.reshape(2,2), x[:,s].reshape(2,1))
#         aug_samp = np.zeros(x.shape)
#         for s in range(aug_samp.shape[1]):
#             aug_samp[:,s] = torch.matmul(torch.tensor(Rotm.reshape(2,2)), x[:,s].reshape(2,1))
    return torch.tensor(aug_samp)



## self-learning contrastive loss
def contrastive_loss(v1, v2):
    tau = 2
    Ls = torch.zeros((len(v1), len(v2)))
    sims_exp = torch.zeros((len(v1), len(v2)))
    for i in range(len(v1)):
        for j in range(len(v2)):
            sims_exp[i][j] = torch.exp(sim(v1[i],v2[j])/tau)
    
    for i in range(len(v1)):
        for j in range(len(v2)):
            num = sims_exp[i,j]
            i_row = sims_exp[i,:]
#             print(num)
            denom = torch.sum(i_row) - sims_exp[i][i]
#             print(denom)
            Ls[i][j] = num/denom
    
    return torch.mean(Ls)
            
            
def sim(a,b):
    return torch.dot(a, b)/(torch.norm(b)*torch.norm(a))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AddGaussianNoise(object):
    def __init__(self, device, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        rand_t = torch.randn(tensor.size())
        rand_t = rand_t.to(device)
        return tensor +  rand_t* self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)