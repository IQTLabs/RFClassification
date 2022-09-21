## all models

from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import pickle
import time
from tqdm import tqdm
import numpy as np
from latency_helpers import *

from sklearn.metrics import ConfusionMatrixDisplay
    

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn.functional as F
from torch.utils.data import Dataset


class PsdSVM():
    def __init__(self, t_seg, nfft, C=1, gamma='scale'):
        self.t_seg = t_seg
        self.nfft = nfft
        self.C = C
        self.gamma = gamma
        self.svc = svm.SVC(kernel='rbf', C=self.C, gamma = self.gamma)
        
    def run_cv(self, X, y, k_fold=5):
        self.cv_acc_ls = []
        self.cv_f1_ls = []
        self.cv_runt_ls = []
        self.cv_models = []
        cv = KFold(n_splits=k_fold, random_state=None, shuffle=True)
        for i, (train_ix, test_ix) in tqdm(enumerate(cv.split(X))):
            svc = svm.SVC(kernel='rbf', C=self.C, gamma = self.gamma)
            svc.fit(X[train_ix], y[train_ix])
            
#             t_start = time.time()
#             y_pred = svc.predict(X[test_ix])
#             t_dur = time.time()-t_start
#             t_batch_avg = t_dur/len(y_pred)
            
                        # predict on the test data (by each sample)
            y_pred, t_indiv = atomic_benchmark_estimator(svc, X[test_ix], output_type=y.dtype, verbose=False)
            t_indiv_avg = np.mean(t_indiv)
            t_inf = t_indiv_avg # inference time
            
            acc = accuracy_score(y[test_ix], y_pred)
            f1 = f1_score(y[test_ix], y_pred, average='weighted')
            
            print('Fold '+str(i+1)+': Accuracy: {:.3},\t F1: {:.3}, \t Runtime: {:.3}'.format(acc,f1, t_inf))
            
            show_confusion_matrix(y[test_ix], y_pred)

            # Add to Lists
            self.cv_runt_ls.append(t_inf)
            self.cv_acc_ls.append(acc)
            self.cv_f1_ls.append(f1)
            self.cv_models.append(svc)

        self.print_result(str(k_fold)+' Fold CV', self.cv_acc_ls, self.cv_f1_ls, self.cv_runt_ls)
        return self.cv_acc_ls, self.cv_f1_ls, self.cv_runt_ls
        
    def run_gridsearch(self, X, y, params, k_fold=5):
        self.grid_best_params_ls = []
        self.grid_acc_ls = []
        self.grid_f1_ls = []
        self.grid_runt_ls = []
        
        kf = KFold(n_splits=k_fold, random_state=None, shuffle=True)

        for i, (train_ix, test_ix) in enumerate(kf.split(X)):

            # find the optimal hypber parameters
            svc = svm.SVC(kernel='rbf', gamma = self.gamma)
            clf = GridSearchCV(svc, params, n_jobs=1)
            clf.fit(X[train_ix], y[train_ix])

            print('Fold '+str(i+1)+' Best Parameters: '+ str(clf.best_params_))
            self.grid_best_params_ls.append(clf.best_params_)

            # predict on the test data
            t_start = time.time()
#             y_pred = clf.predict(X[test_ix])
            
            y_pred, runtimes = atomic_benchmark_estimator(clf, X[test_ix], y.dtype, verbose=False) # predict & measure time
        
#             print(y_pred)
#             
            acc = accuracy_score(y[test_ix], y_pred)
#             print(y_pred)
#             print(y[test_ix])
            f1 = f1_score(y[test_ix], y_pred, average='weighted')
            print('Fold '+str(i+1)+': Accuracy: {:.3},\t F1: {:.3}, \t Runtime: {:.3}'.format(acc,f1, np.mean(runtimes)))
            
            # Append to List
            self.grid_acc_ls.append(acc)
            self.grid_f1_ls.append(f1)
            self.grid_runt_ls.append(np.mean(runtimes))
        
        self.print_result(str(k_fold)+' Fold GridSearch CV', self.grid_acc_ls, self.grid_f1_ls, self.grid_runt_ls)
        return self.grid_acc_ls, self.grid_f1_ls, self.grid_runt_ls, self.grid_best_params_ls
        
    
    def train(self, X_train, y_train):
        self.svc.fit(X_train, y_train)
        
    def predict(self, X):
        return self.svc.predict(X)
        
        
    def save(self, model_path, model_name):
        print('Model saved as:', model_path+model_name)
        pickle.dump(self.svc, open(model_path+model_name, 'wb'))
        
    def print_result(self, test_name, acc_ls, f1_ls, runt_ls):
        out_msg = 'PSD+SVM '+ test_name+ ' acc: {:.3}, F1: {:.3}, Run-time: {:.3}ms'.format(np.mean(acc_ls), np.mean(f1_ls), np.mean(runt_ls)*1e3)
        print(out_msg)
        
        
        
##### Visualization of Models #####
def show_confusion_matrix(labels, predictions, normalize='true', to_save=False, plot_folder='na', plot_name='na'):
    disp = ConfusionMatrixDisplay.from_predictions(labels, predictions, normalize=normalize)
#     display_labels=list(dataset.class_to_idx.keys())
    if to_save:
        disp.figure_.savefig(f"{plot_folder}/confusion_matrix_{plot_name}.png")



### VGG features with Fully Connected Layer
class VGGFC(nn.Module):
    def __init__(self, num_classes, isarray=False):
        super(VGGFC,self).__init__()
        self.num_classes = num_classes
        self.isarray = isarray
        self.vggfull = models.vgg16(pretrained=True)
        modules=list(self.vggfull.children())[:-1] # remove the fully connected layer & adaptive averaging
        self.vggfeats=nn.Sequential(*modules)
        
        for param in self.vggfeats.parameters():
            param.requires_grad_(False)
        
        self._fc = nn.Linear(25088, num_classes)
    def forward(self, x):
        if self.isarray:
            x = torch.unsqueeze(x, 1)
            x = x.repeat(1,3,1,1)
        else:
            if len(x.shape)==4:
                x = torch.moveaxis(x,-1, 1) # move from (225,225,3) to (3,225,225) 
            else:
                x = torch.moveaxis(x, -1, 0)
        x = self.vggfeats(x)
        x = x.reshape(-1,25088)
        x = self._fc(x)
        
        return x
    
    def reset_weights(self):
        print(f'Reset trainable parameters of layer = {self._fc}')
        self._fc.reset_parameters()

class ResNetFC(nn.Module):
    def __init__(self, num_classes):
        super(ResNetFC,self).__init__()
        self.num_classes = num_classes
        self.resnetfull = models.resnet50(pretrained=True)
        modules=list(self.resnetfull.children())[:-2] # remove the fully connected layer & adaptive averaging
        self.resnetfeats=nn.Sequential(*modules)
        
        for param in self.resnetfeats.parameters():
            param.requires_grad_(False)
        
        self._fc = nn.Linear(100352, num_classes)
    def forward(self, x):
        if len(x.shape)==4:
            x = torch.moveaxis(x,-1, 1)
        else:
            x = torch.moveaxis(x, -1, 0)
        x = self.resnetfeats(x)
        x = x.reshape(-1,100352)
        x = self._fc(x)
        
        return x
    
    def reset_weights(self):
        print(f'Reset trainable parameters of layer = {self._fc}')
        self._fc.reset_parameters()
    

class RFUAVNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(RFUAVNet, self).__init__()
        self.num_classes = num_classes

        self.dense = nn.Linear(320, num_classes) #320 inputs in original paper with 0.25ms input
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.smax = nn.Softmax(dim=1)
        
        # for r unit
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, stride=5, dtype=torch.float32)
        self.norm1 = nn.BatchNorm1d(num_features=64, dtype=torch.float32)
        self.elu1 = nn.ELU(alpha=1.0, inplace=False)
        
        # setup for components of the gunit
        self.groupconvlist = []
        self.norm2list = []
        self.elu2list = []
        for i in range(4):
            self.groupconvlist.append( nn.Conv1d( 
                  in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  stride = 2,
                  groups=8,
    #               bias=False,
                  dtype=torch.float32
                ))
            self.norm2list.append(nn.BatchNorm1d(num_features=64))
            self.elu2list.append(nn.ELU(alpha=1.0, inplace=False))
        self.groupconv = nn.ModuleList(self.groupconvlist)
        self.norm2 = nn.ModuleList(self.norm2list)
        self.elu2 = nn.ModuleList(self.elu2list)
        
        # multi-gap implementation
        self.avgpool1000 = nn.AvgPool1d(kernel_size=1000)
        self.avgpool500 = nn.AvgPool1d(kernel_size=500)
        self.avgpool250 = nn.AvgPool1d(kernel_size=250)
        self.avgpool125 = nn.AvgPool1d(kernel_size=125)
    
    # Progresses data across layers    
    def forward(self, x):
        # runit first
        x1 = self.runit(x)
        xg1 = self.gunit(F.pad(x1, (1,0)), 0) 
        x2 = self.pool(x1)
        x3 = xg1+x2
        
        # series of gunits
        xg2 = self.gunit(F.pad(x3, (1,0)), 1)
        x4 = self.pool(x3)
        x5 = xg2+x4
        
        xg3 = self.gunit(F.pad(x5, (1,0)), 2)
        x6 = self.pool(x5)
        x7 = x6+xg3
        
        xg4 = self.gunit(F.pad(x7, (1,0)), 3)
        x8 = self.pool(x7)
        x_togap = x8+xg4
        
        
        # gap and multi-gap
        f_gap_1 = self.avgpool1000(xg1)
        f_gap_2 = self.avgpool500(xg2)
        f_gap_3 = self.avgpool250(xg3)
        f_gap_4 = self.avgpool125(xg4)
        
        f_multigap = torch.cat((f_gap_1,f_gap_2, f_gap_3, f_gap_4), 1)
        
        f_gap_add = self.avgpool125(x_togap)
    
        f_final = torch.cat((f_multigap, f_gap_add),1)
        f_flat = f_final.flatten(start_dim=1)
    
        out = self.dense(f_flat)
#         out = self.smax(f_fc)
        # fc_layer
        
        return out
    
    def runit(self, x):
#         print('r unit input typpe', x.dtype)
#         print(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.elu1(x)
        return x
        
    def gunit(self, x, n):
        # group convolution layer 8 by 8
        # norm
        # elu
        # n indicates which gunit
        x = self.groupconv[n](x) 
        x = self.norm2[n](x)
        x = self.elu2[n](x)
        return x
    
    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()
            elif isinstance(layer, nn.ModuleList):
                for item in layer.children():
                    if hasattr(item, 'reset_parameters'):
                        print(f'Reset trainable parameters of layer = {item}')
                        item.reset_parameters()
   
        