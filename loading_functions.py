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

### 1. DRONEDETECT Dataset ###

# functions for loading data and pytorch dataset classes
class DroneDetectTorch(Dataset):
    def __init__(self, feat_folder, feat_name, seg_len, n_per_seg, feat_format, output_feat, interferences):
        self.feat_format = feat_format
        self.output_feat = output_feat
        self.dir_name = feat_folder+feat_format+'_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
        print('Directory Name: ', self.dir_name)
        files = os.listdir(self.dir_name)
        files = [fi for fi in files if is_interference(fi, interferences)]
        self.files = files
        
        unique_labels = sorted(list(set([self.get_label(fi, self.output_feat) for fi in self.files])))
        self.unique_labels = unique_labels
        self.class_to_idx = {lbl:i for i,lbl in enumerate(self.unique_labels)}
        self.idx_to_class = {i:lbl for i,lbl in enumerate(self.unique_labels)}
        
        # get the length of each of the files (multiple samples in each file
        self.fi_lens = np.zeros(len(self.files))
        if self.feat_format == 'ARR':
            for i, fi in enumerate(self.files):
                DATA = np.load(self.dir_name+self.files[i], allow_pickle=True).item()['feat']
                try:
                    self.fi_lens[i] = self.fi_lens[i-1]+len(DATA)  
                except:
                    self.fi_lens[i] = len(DATA)
            self.fi_lens = self.fi_lens.astype(int)
       
        # print data shape
        print('dataset size', len(self))
        print('shape of each item', self.__getitem__(0)[0].shape)
            
        
    def __len__(self):
        if self.feat_format == 'IMG':
            return len(self.files) # one image per file
        else:
            return self.fi_lens[-1]
    
    def __getitem__(self, i):
        if not isinstance(i, list):
            # if single integer
            return self.__getitemsingle__(i)
        ft_ls = []
        lb_ls = []
        for ii in i:
            ft, lb = self.__getitemsingle__(ii)
            ft_ls.append(ft)
            lb_ls.append(lb)
        
        return np.array(ft_ls), np.array(lb_ls)
    
    def __getitemsingle__(self, i):
        # all data must be in float and tensor format
        
        if self.feat_format == 'ARR':
            # convert i to file number and index within file
            i_file = np.argwhere(self.fi_lens>i)[0][0]
            DATA = np.load(self.dir_name+self.files[i_file], allow_pickle=True).item()            
            i_infile = int(len(DATA['feat'])- (self.fi_lens[i_file]-i))
            Feat = DATA['feat'][i_infile]
            # apply norm
            Feat = Feat/np.max(Feat)
            Label = DATA[self.output_feat][i_infile]
            #Label = Label.reshape(len(Label),1)
        elif self.feat_format == 'IMG':
            DATA = cv2.imread(self.dir_name+self.files[i])
            DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2RGB)
            Feat = DATA/255
            Feat = torch.tensor(Feat).float()
            Label = self.class_to_idx[self.get_label(self.files[i], self.output_feat)]
                
        return Feat,Label

    def get_arrays(self):
        i_all = list(range(len(self)))
        X_use, y_use = self.__getitem__(i_all)
        return X_use, y_use
    
    # return the label based on file name
    # image file names saved in DRONE+'_'+COND+'_'+INT+'_'+FIn+'_'+str(counter)+'.jpg' format
    # in array format: COND_FEAT_DRONE_MODE_nFFT
    def get_label(self, filename, label_name):
        if self.feat_format == 'IMG':
            if label_name == 'drones':
                return filename[:3]
            if label_name == 'conds':
                return filename[4:6]
            if label_name == 'ints':
                return filename[7:9]
        elif self.feat_format == 'ARR':
            name_parts = filename.split('_')
            if label_name == 'drones':
                return name_parts[2]
            if label_name == 'conds':
                return name_parts[3]
            if label_name == 'ints':
                return name_parts[0]
        return None

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
            if file_name.split('_')[2] == interference_map[f] or file_name.find(f)!=-1: 
#         if file_name.find(f)!=-1:
                return True
        except:
            pass
    
    return False

            

### 2. DroneRF DATASET ### 
class DroneRFTorch(Dataset):
#     feat_folder, feat_name, seg_len, n_per_seg, highlow, output_feat
     def __init__(self, feat_folder, feat_name, seg_len, n_per_seg, feat_format, output_feat, highlow):
        self.feat_format = feat_format
        self.output_feat = output_feat
        sub_folder_name = feat_format+'_'+feat_name+'_'+highlow+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
        self.dir_name = feat_folder+sub_folder_name
        print(self.dir_name)
        self.files = os.listdir(self.dir_name)
        
        # do we need a unique label
#         unique_labels = sorted(list(set([self.get_label(fi, self.output_feat) for fi in self.files])))
#         self.unique_labels = unique_labels
        
        # convert output drone codes to names
        if self.output_feat == 'drones':
            self.lbl_to_class = {0:"None", 101:"AR", 100:"Bebop", 110:"Phantom"}
        elif self.output_feat == 'modes':
            self.lbl_to_class = {0:"None", 
                                 10000:"Bebop1", 10001:"Bebop2", 10010:"Bebop3", 10011:"Bebop4",
                                10100:"AR1", 10101:"AR2", 10110:"AR3", 10111:"AR4",
                                11000:"Phantom1", 11001:"Phantom2", 11010:"Phantom3", 11011:"Phantom4"}
        elif self.output_feat =='bi':
            self.lbl_to_class = {0:"None", 1:"Drone"}
            
        self.class_to_idx = {lbl:i for i,lbl in enumerate(self.lbl_to_class)}
        self.idx_to_class = {i:lbl for i,lbl in enumerate(self.lbl_to_class)}

     # get the length of each of the files (multiple samples in each file
        self.fi_lens = np.zeros(len(self.files))
        if self.feat_format == 'ARR':
            for i, fi in enumerate(self.files):
                DATA = np.load(self.dir_name+self.files[i], allow_pickle=True).item()['feat']
                try:
                    self.fi_lens[i] = self.fi_lens[i-1]+len(DATA)  
                except:
                    self.fi_lens[i] = len(DATA)
            self.fi_lens = self.fi_lens.astype(int)

        # print data shape
        print('dataset size', len(self))
        print('shape of each item', self.__getitem__(0)[0].shape)
#         print('test')

     def __len__(self):
        if self.feat_format == 'IMG':
            return len(self.files) # one image per file
        else:
            return self.fi_lens[-1] # multiple samples per file - get the last of the lengths
    
     def __getitem__(self, i):
        if not isinstance(i, list):
            # if single integer
            return self.__getitemsingle__(i)
        ft_ls = []
        lb_ls = []
        for ii in i:
            ft, lb = self.__getitemsingle__(ii)
            ft_ls.append(ft)
            lb_ls.append(lb)
            
        return np.array(ft_ls), np.array(lb_ls)
    
     def __getitemsingle__(self, i):
        if self.feat_format == 'ARR':
            # convert i to file number and index within file
            i_file = np.argwhere(self.fi_lens>i)[0][0]
            DATA = np.load(self.dir_name+self.files[i_file], allow_pickle=True).item()            
            i_infile = int(len(DATA['feat'])- (self.fi_lens[i_file]-i))
            Feat = DATA['feat'][i_infile]
            # apply norm
            Feat = Feat/np.max(Feat)
            Label = DATA[self.output_feat][i_infile]
            Label = self.lbl_to_class[Label]
                
        elif self.feat_format == 'IMG':
            DATA = cv2.imread(self.dir_name+self.files[i])
            DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2RGB)
            Feat = DATA/255
            Feat = torch.tensor(Feat).float()
            Label = self.lbl_to_class[self.get_label(self.files[i], self.output_feat)]
                
        return Feat,Label
        
    # return all data at location
     def get_arrays(self):
        i_all = list(range(len(self)))
        X_use, y_use = self.__getitem__(i_all)
        return X_use, y_use
    
     def get_label(self, fi, label_name):
            bui = fi.split('_')[0]
            if label_name == 'bi':
                return int(bui[0])
            if label_name == 'drones':
                return int(bui[:3])
            if label_name == 'modes':
                return int(bui[:5])
            
            
#      def get_label_drones(self, filename, label_name):
#         if self.feat_format == 'IMG':
#             if label_name == 'drones':
#                 return filename[:3]
#             if label_name == 'conds':
#                 return filename[4:6]
#             if label_name == 'ints':
#                 return filename[7:9]
#         elif self.feat_format == 'ARR':
#             name_parts = filename.split('_')
#             if label_name == 'drones':
#                 return name_parts[2]
#             if label_name == 'conds':
#                 return name_parts[3]
#             if label_name == 'ints':
#                 return name_parts[0]
#         return None

    
    
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
            # not sure why one pair of files have different lengths (skip this file for now)
        else:
            # stack the features and ys
            rf_sig = np.vstack((rf_data_h, rf_data_l))

            # decide on segment lengths
            len_seg = int(t_seg/1e3*fs)
            n_segs = (len(rf_data_h))//len_seg
            n_keep = n_segs*len_seg
            try:
                rf_sig = np.split(rf_sig[:,:n_keep], n_segs, axis =1) # samples of 1e4
            except:
                print('error on splitting')
                return rf_sig, n_keep, n_segs, len_seg
            Xs.append(normalize_rf(rf_sig))

            y_rep = np.repeat(int(low_freq_files[i][0]),n_segs)
            y4_rep = np.repeat(int(low_freq_files[i][:3]),n_segs)
            y10_rep = np.repeat(int(low_freq_files[i][:5]),n_segs)

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

### 3. GamutRF scanner collected Data ###

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


    
####~ARCHIVE FUNCTIONS###
## Load numerical feature files
# Inputs:
    # - feat_folder: where the files are located
    # - feat_name: choose from 'PSD', 'SPEC'
    # - interferences: choose from ['BLUE', 'WIFI', 'BOTH', 'CLEAN']
    # - nperseg: n per segment, part of the file names
    # - datestr: date feature files were generated
    
# def load_dronedetect_features(feat_folder, feat_name, seg_len, n_per_seg, feat_format, label_name, interferences):
#     sub_folder_name = feat_format+'_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
#     augmented_int = aug_int(interferences) # map interference names to codes
#     files = os.listdir(feat_folder+sub_folder_name)
#     for i in tqdm(range(len(files))):
#         fi = files[i]
        
#         if is_interference(fi, interferences):
#             if feat_format == 'ARR':
#                 DATA = np.load(feat_folder+sub_folder_name+fi, allow_pickle=True).item()
#                 Feat = DATA['feat']
#                 Label = DATA[label_name]
#                 Label = Label.reshape(len(Label),1)
#             elif feat_format == 'IMG':
#                 DATA = cv2.imread(feat_folder+sub_folder_name+fi)
#                 Feat = DATA
#                 Feat = np.expand_dims(Feat, axis=0)
#                 Label = get_label(fi, label_name)
#                 Label = np.array(Label)
# #             print(Feat.shape)
#             try:
#                 Xs_arr = np.concatenate((Xs_arr, Feat),axis=0)
# #                 print('concatenated')
#                 y_arr = np.vstack((y_arr, Label))
# #                 print('stacked')
#             except: # if the files have not been created yet
#                 Xs_temp = Feat
#                 y_temp = Label
# #                 if Xs_temp.shape[0]>5 and y_temp.shape[0]>5: # some files are not properly saved (omit for now)
#                 Xs_arr = Xs_temp
#                 y_arr = y_temp
   
#     return Xs_arr, y_arr




# def load_drone_detect_images(feat_folder, feat_name, seg_len, n_per_seg, output_feat, interferences):
#     sub_folder_name = 'IMG_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
    
#     files = os.listdir(feat_folder+sub_folder_name)
#     for i in tqdm(range(len(files))):
#         fi = files[i]
        
#         if is_interference(fi, interferences):
#             img = cv2.imread(feat_folder+sub_folder_name+fi)
