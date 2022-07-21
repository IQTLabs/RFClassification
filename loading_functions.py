from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
from helper_functions import *
from sklearn import preprocessing
from sklearn.metrics import f1_score
import torch
import pandas as pd
import cv2

# functions for loading data and pytorch dataset classes

class DroneDetectTorch(Dataset): ## NUMBERICAL DATA
    def __init__(self, feat_folder, feat_name, seg_len, n_per_seg, feat_format, output_feat, interferences):
        self.feat_format = feat_format
        self.output_feat = output_feat
        self.dir_name = feat_folder+feat_format+'_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
        files = os.listdir(self.dir_name)
        files = [fi for fi in files if is_interference(fi, interferences)]
        self.files = files
        
        unique_labels = sorted(list(set([get_label(fi, self.output_feat) for fi in self.files])))
        self.unique_labels = unique_labels
        self.class_to_idx = {lbl:i for i,lbl in enumerate(self.unique_labels)}
        self.idx_to_class = {i:lbl for i,lbl in enumerate(self.unique_labels)}
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        # all data must be in float and tensor format
        
        if self.feat_format == 'ARR':
            DATA = np.load(self.dir_name+self.files[i], allow_pickle=True).item()
            Feat = DATA['feat']
            Label = DATA[self.output_feat]
            #Label = Label.reshape(len(Label),1)
        elif self.feat_format == 'IMG':
            DATA = cv2.imread(self.dir_name+self.files[i])
            DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2RGB)
            Feat = DATA/255
            #Feat = np.expand_dims(Feat, axis=0)
            Label = self.class_to_idx[get_label(self.files[i], self.output_feat)]
            #Label = np.array(Label)
                
        return Feat,Label
    

## Create a dataset class
## Creating a custom dataset
class DroneData(Dataset): ## NUMBERICAL DATA
    def __init__(self, Xarr, yarr):
        self.Xarr = Xarr
        self.le = preprocessing.LabelEncoder()
        self.le.fit(yarr.flatten())
        self.yarr = self.le.transform(yarr.flatten())
        
    def __len__(self):
        return len(self.yarr)
    
    def __getitem__(self, index):
        # all data must be in float and tensor format
        X = torch.tensor((self.Xarr[index]))/255
        X = X.type(torch.float)
#         X = X.unsqueeze(0) # why
        y = torch.tensor((self.yarr[index]))
        return (X, y)
    
# Load data for Drone Detect (original file from authors)
# input: file_path
#        t_seg: duration of the segment in miliseconds
def load_dronedetect_raw(file_path, t_seg):
    fs = 60e6 #60 MHz
    f = open(file_path, "rb")                                        # open file
    data = np.fromfile(f, dtype="float32",count=240000000)      # read the data into numpy array
    data = data.astype(np.float32).view(np.complex64)           # view as complex
    data_norm = (data-np.mean(data))/(np.sqrt(np.var(data)))    # normalise
    # decide on segment lengths
    len_seg = int(t_seg/1e3*fs)
    n_segs = (len(data_norm))//len_seg
    n_keep = n_segs*len_seg
    newarr = np.array_split(data_norm[:n_keep], n_segs)                  # split the array, 100 will equate to a sample length of 20ms
    # 10 Splits into 200ms chunks
    return newarr, data_norm
    

def load_dronedetect_data(feat_folder, feat_name, seg_len, n_per_seg, feat_format, output_feat, interferences):
# A loading function to return a dataset variable '''
    Xs_arr, y_arr = load_dronedetect_features(feat_folder, feat_name, seg_len, n_per_seg, feat_format, output_feat, interferences)
    dataset = DroneData(Xs_arr, y_arr)
    return dataset


## Load numerical feature files
# Inputs:
    # - feat_folder: where the files are located
    # - feat_name: choose from 'PSD', 'SPEC'
    # - interferences: choose from ['BLUE', 'WIFI', 'BOTH', 'CLEAN']
    # - nperseg: n per segment, part of the file names
    # - datestr: date feature files were generated
    
def load_dronedetect_features(feat_folder, feat_name, seg_len, n_per_seg, feat_format, label_name, interferences):
    print(interferences)
    sub_folder_name = feat_format+'_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
    #augmented_int = aug_int(interferences) # map interference names to codes
    files = os.listdir(feat_folder+sub_folder_name)
    for i in tqdm(range(len(files))):
        fi = files[i]

        if is_interference(fi, interferences):
            print(files[i])
            if feat_format == 'ARR':
                DATA = np.load(feat_folder+sub_folder_name+fi, allow_pickle=True).item()
                Feat = DATA['feat']
                Label = DATA[label_name]
                Label = Label.reshape(len(Label),1)
            elif feat_format == 'IMG':
                DATA = cv2.imread(feat_folder+sub_folder_name+fi)
                DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2RGB)
                Feat = DATA
                Feat = np.expand_dims(Feat, axis=0)
                Label = get_label(fi, label_name)
                Label = np.array(Label)
#             print(Feat.shape)
            try:
                Xs_arr = np.concatenate((Xs_arr, Feat),axis=0)
#                 print('concatenated')
                y_arr = np.vstack((y_arr, Label))
#                 print('stacked')


            except: # if the files have not been created yet
                Xs_temp = Feat
                y_temp = Label
#                 if Xs_temp.shape[0]>5 and y_temp.shape[0]>5: # some files are not properly saved (omit for now)
                Xs_arr = Xs_temp
                y_arr = y_temp
            
            
   
    return Xs_arr, y_arr

# return the label based on file name
# image file names saved in DRONE+'_'+COND+'_'+INT+'_'+FIn+'_'+str(counter)+'.jpg' format
def get_label(filename, label_name):
    if label_name == 'drones':
        return filename[:3]
    if label_name == 'conds':
        return filename[4:6]
    if label_name == 'ints':
        return filename[7:9]
    return None
    
def aug_int(interferences):
    
#     00 for a clean signal, 01 for Bluetooth only, 10 for Wi-Fi only and 11 for Bluetooth and Wi-Fi interference concurrently.
    if 'CLEAN' in interferences:
        interferences.append('00')
    if 'BLUE' in interferences:
        interferences.append('01')
    if 'WIFI' in interferences:
        interferences.append('10')
    if 'BOTH' in interferences:
        interferences.append('11')
    return interferences


interference_map = {'CLEAN':'00', 'BLUE':'01', 'WIFI':'10', 'BOTH':'11'}

def is_interference(file_name, int_list):
    
    for f in int_list:
        try: 
            if file_name.split('_')[2] == interference_map[f]: 
            #if file_name.find(interference_map[f])!=-1:
                
                return True
        except: 
            pass
    
    return False


# def load_drone_detect_images(feat_folder, feat_name, seg_len, n_per_seg, output_feat, interferences):
#     sub_folder_name = 'IMG_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
    
#     files = os.listdir(feat_folder+sub_folder_name)
#     for i in tqdm(range(len(files))):
#         fi = files[i]
        
#         if is_interference(fi, interferences):
#             img = cv2.imread(feat_folder+sub_folder_name+fi)
            

# function to load drone rf data raw in array form
def load_dronerf_raw(main_folder, t_seg):
    high_freq_files = os.listdir(main_folder+'High/')
    low_freq_files = os.listdir(main_folder+'Low/')

    high_freq_files.sort()
    low_freq_files.sort()
    fs = 40e6 #40 MHz

    # feature & results lists
    Xs = []
    ys = []
    y4s = []
    y10s = []

    for i in range(len(high_freq_files)):
        # load RF data
        rf_data_h = pd.read_csv(main_folder+'High/'+high_freq_files[i], header=None).values
        rf_data_h = rf_data_h.flatten()

        rf_data_l = pd.read_csv(main_folder+'Low/'+low_freq_files[i], header=None).values
        rf_data_l = rf_data_l.flatten()

        if len(rf_data_h)!=len(rf_data_l):
            print('diff', i, 'file name:', low_freq_files[i]) 
            # not sure why one pair of files have different lengths (ignore this for now)
        else:
            # stack the features and ys
            rf_sig = np.vstack((rf_data_h, rf_data_l))

            # decide on segment lengths
            len_seg = int(t_seg/1e3*fs)
            n_segs = (len(rf_data_h))//len_seg
#             print('len of full file:', len(rf_data_h))
#             print('len sig:', len_seg)
            n_keep = n_segs*len_seg

            rf_sig = np.split(rf_sig[:n_keep], n_segs, axis =1) # samples of 1e4
            Xs.append(normalize_rf(rf_sig))

            y_rep = np.repeat(int(low_freq_files[i][0]),1000)
            y4_rep = np.repeat(int(low_freq_files[i][:3]),1000)
            y10_rep = np.repeat(int(low_freq_files[i][:5]),1000)

            ys.append(y_rep) # 2 class
            y4s.append(y4_rep) # 4 class
            y10s.append(y10_rep) # 10 class

            if int(high_freq_files[i][:5])!= int(low_freq_files[i][:5]):
                raise Exception("File labels do not match")

    # shape the arrays
    Xs_arr = np.array(Xs)
    Xs_arr = Xs_arr.reshape(-1, *Xs_arr.shape[-2:])
    ys_arr = np.array(ys).flatten()
    y4s_arr = np.array(y4s).flatten()
    y10s_arr = np.array(y10s).flatten()
    return Xs_arr, ys_arr, y4s_arr, y10s_arr

def load_dronerf_features(feat_folder, feat_name, seg_len, n_per_seg, highlow, output_label):
    sub_folder_name = 'ARR_'+feat_name+'_'+highlow+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
    
    files = os.listdir(feat_folder+sub_folder_name)
    for i in tqdm(range(len(files))):
        fi = files[i]
        DATA = np.load(feat_folder+sub_folder_name+fi, allow_pickle=True).item()
        try:
            Xs_arr = np.concatenate((Xs_arr, DATA['feat']),axis=0)
#                 print('concatenated')
            y_arr = np.vstack((y_arr, DATA[output_label].reshape(len(DATA[output_label]),1)))
        except: # if the files have not been created yet
            Xs_temp = DATA['feat']
            y_temp = DATA[output_label].reshape(len(DATA[output_label]),1)
            if Xs_temp.shape[0]>5 and y_temp.shape[0]>5: # some files are not properly saved (omit for now)
                Xs_arr = Xs_temp
                y_arr = y_temp
    
    return Xs_arr, y_arr

# load the generated features from gamut collection day
def load_gamut_features(data_path, feat_name):
    files = os.listdir(data_path)
    for i in tqdm(range(len(files))):
        fi = files[i]
        if feat_name not in fi:
            continue
        DATA = np.load(data_path+fi, allow_pickle=True).item()
        try:
            Xs_arr = np.concatenate((Xs_arr, DATA['feat']),axis=0)
#                 print('concatenated')
#             y_arr = np.vstack((y_arr, DATA[output_label].reshape(len(DATA[output_label]),1)))
        except: # if the files have not been created yet
            Xs_temp = DATA['feat']
#             y_temp = DATA[output_label].reshape(len(DATA[output_label]),1)
            if Xs_temp.shape[0]>5: # some files are not properly saved (omit for now)
                Xs_arr = Xs_temp
#                 y_arr = y_temp
    
    return Xs_arr
