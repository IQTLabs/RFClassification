from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
# from helper_functions import *
from sklearn import preprocessing
import torch

# functions for loading data and pytorch dataset classes


## Create a dataset class
## Creating a custom dataset
class DroneDetectData(Dataset): ## NUMBERICAL DATA
    def __init__(self, Xarr, yarr):
        self.Xarr = Xarr
        test_list=[]
        self.le = preprocessing.LabelEncoder()
        self.le.fit(yarr.flatten())
        self.yarr = self.le.transform(yarr.flatten())
        
    def __len__(self):
        return len(self.yarr)
    
    def __getitem__(self, index):
        # all data must be in float and tensor format
        X = torch.tensor((self.Xarr[index]))
        X = X.unsqueeze(0)
        y = torch.tensor(float(self.yarr[index]))
        return (X, y)
    

def load_dronedetect_data(feat_folder, feat_name, seg_len, n_per_seg, interferences):
# A loading function to return a dataset variable '''
    Xs_arr, y_arr = load_features_arr(feat_folder, feat_name, seg_len, n_per_seg, interferences)
            
    dataset = DroneDetectData(Xs_arr, y_arr)
    return dataset


## Load numerical feature files
# Inputs:
    # - feat_folder: where the files are located
    # - feat_name: choose from 'PSD', 'SPEC'
    # - interferences: choose from ['BLUE', 'WIFI', 'BOTH', 'CLEAN']
    # - nperseg: n per segment, part of the file names
    # - datestr: date feature files were generated
    
def load_features_arr(feat_folder, feat_name, seg_len, n_per_seg, interferences):
    sub_folder_name = 'ARR_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
    
    files = os.listdir(feat_folder+sub_folder_name)
    for i in tqdm(range(len(files))):
        fi = files[i]
        
        if is_interference(fi, interferences):
            DATA = np.load(feat_folder+sub_folder_name+fi, allow_pickle=True).item()
            try:
                Xs_arr = np.concatenate((Xs_arr, DATA['feat']),axis=0)
#                 print('concatenated')
                y_arr = np.vstack((y_arr, DATA['drones'].reshape(len(DATA['drones']),1)))
            except: # if the files have not been created yet
                Xs_temp = DATA['feat']
                y_temp = DATA['drones'].reshape(len(DATA['drones']),1)
                if Xs_temp.shape[0]>5 and y_temp.shape[0]>5: # some files are not properly saved (omit for now)
                    Xs_arr = Xs_temp
                    y_arr = y_temp
            
#             print(Xs_arr.shape)
   
    return Xs_arr, y_arr

def is_interference(file_name, int_list):
    for f in int_list:
        if file_name.find(f)!=-1:
            return True
    
    return False