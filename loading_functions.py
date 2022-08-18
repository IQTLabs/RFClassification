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
import time
from scipy import interpolate

### 1. DRONEDETECT Dataset ###

# functions for loading data and pytorch dataset classes
class DroneDetectTorch(Dataset):
    def __init__(self, feat_folder, feat_name, seg_len, n_per_seg, feat_format, output_feat, output_tensor, interferences):
        self.feat_format = feat_format
        self.output_feat = output_feat
        self.output_tensor = output_tensor
        self.feat_name = feat_name
        
        if feat_name !='RAW':
            self.dir_name = feat_folder+feat_format+'_'+feat_name+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
        else:
            self.dir_name = feat_folder+feat_format+'_'+feat_name+'_'+str(10000)+'_'+str(seg_len)+'/'
        print('Directory Name: ', self.dir_name)
        files = os.listdir(self.dir_name)
        files = [fi for fi in files if is_interference(fi, interferences)]
        self.files = files
#         print(files)
        
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
            if self.feat_name == 'SPEC':
                # conver to dB
                Feat = -10*np.log10(Feat)
            # apply norm
            Feat = Feat/np.max(Feat)
            Label = DATA[self.output_feat][i_infile]
            #Label = Label.reshape(len(Label),1)
        elif self.feat_format == 'IMG':
            DATA = cv2.imread(self.dir_name+self.files[i])
            DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2RGB)
#             DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2GRAY ) # change to grayscale
            Feat = DATA/255
            Label =self.get_label(self.files[i], self.output_feat)
        
        if self.output_tensor:
            Feat = torch.tensor(Feat).float()
            Label = self.class_to_idx[Label] # also use numerical labels
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
     def __init__(self, feat_folder, feat_name, seg_len, n_per_seg, feat_format, output_feat, output_tensor, highlow, to_norm=False):
        self.feat_format = feat_format
        self.output_feat = output_feat
        self.output_tensor = output_tensor
        self.feat_name = feat_name
        self.to_norm = to_norm

        if feat_name == 'RAW':
            sub_folder_name = feat_format+'_'+feat_name+'_'+str(10000)+'_'+str(seg_len)+'/'
        else:
            sub_folder_name = feat_format+'_'+feat_name+'_'+highlow+'_'+str(n_per_seg)+'_'+str(seg_len)+'/'
        self.dir_name = feat_folder+sub_folder_name
        print(self.dir_name)
        self.files = os.listdir(self.dir_name)
        
        # do we need a unique label
#         unique_labels = sorted(list(set([self.get_label(fi, self.output_feat) for fi in self.files])))
#         self.unique_labels = unique_labels
        
        # convert output drone codes to names
        if self.output_feat == 'drones':
            self.bui_to_class = {0:"None", 101:"AR", 100:"Bebop", 110:"Phantom"}
        elif self.output_feat == 'modes':
            self.bui_to_class = {0:"None", 
                                 10000:"Bebop1", 10001:"Bebop2", 10010:"Bebop3", 10011:"Bebop4",
                                10100:"AR1", 10101:"AR2", 10110:"AR3", 10111:"AR4",
                                11000:"Phantom1", 11001:"Phantom2", 11010:"Phantom3", 11011:"Phantom4"}
        elif self.output_feat =='bi':
            self.bui_to_class = {0:"None", 1:"Drone"}
            
        self.class_to_idx = {lbl:i for i,lbl in enumerate(self.bui_to_class.values())}
        self.idx_to_class = {i:lbl for i,lbl in enumerate(self.bui_to_class.values())}
        self.unique_labels = list(self.bui_to_class.values())

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
            if self.to_norm:
                Feat = Feat/np.max(Feat)
            Label = DATA[self.output_feat][i_infile]
            Label = self.bui_to_class[Label]
                
        elif self.feat_format == 'IMG':
            DATA = cv2.imread(self.dir_name+self.files[i])
            DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2RGB)
            Feat = DATA/255
            Label = self.bui_to_class[self.get_bui(self.files[i], self.output_feat)]
        
        if self.output_tensor:
            Feat = torch.tensor(Feat).float()
            Label = self.class_to_idx[Label]
                
        return Feat,Label
        
    # return all data at location
     def get_arrays(self):
        i_all = list(range(len(self)))
        X_use, y_use = self.__getitem__(i_all)
        return X_use, y_use
    
     def get_bui(self, fi, label_name):
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
            Xs.append(rf_sig)

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

## method to normalize for rfuav-net - *Note in original paper - they normalized across all drones
def normalize_rf(rf):
    """apply normalization to data in the numpy array format"""
    rfnorm = []
    for i in range(len(rf)): # for each segment
        rfnorm_i = np.zeros(rf[i].shape)
        for j in range(2):
            r = (np.max(rf[i][j])-np.min(rf[i][j]))
            m = np.min(rf[i][j])
            rfnorm_i[j] = (rf[i][j]-m)/r
        rfnorm.append(rfnorm_i)

    return rfnorm

def load_dronerf_raw_single(file_path, t_seg):
#     print('loading from:', file_path)
    rf_data = pd.read_csv(file_path, header=None).values
    rf_data = rf_data.flatten()
    fs = 40e6 #40 MHz
    # decide on segment lengths
    len_seg = int(t_seg/1e3*fs)
    n_segs = (len(rf_data))//len_seg
    n_keep = n_segs*len_seg
#     print('n_segs:', n_segs, ' length of each seg:', len_seg)
    rf_sig = np.split(rf_data[:n_keep], n_segs, axis =0)
#     rf_sig = np.split(rf_sig[:,:n_keep], n_segs, axis =1) # samples of 1e4

    
    return rf_sig

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


#### 4. Dataloaders for RAW dataset data (IQ and magnitude)
class RFRawTorch(Dataset):
    def __init__(self, main_folder, t_seg, output_feat, whichdata, downsamp, interferences=None):
        self.t_seg = t_seg
        self.output_feat = output_feat
        self.interference = interferences
        self.whichdata = whichdata
        self.downsamp = downsamp
        self.files = []
        
        if whichdata =='dronerf':
            self.files_h = os.listdir(main_folder+'High/')
            self.files_l = os.listdir(main_folder+'Low/')
            
            self.files_h = [main_folder+'High/'+ff for ff in self.files_h]
            self.files_l = [main_folder+'Low/'+ff for ff in self.files_l]
            
            self.fi_lens = np.zeros(len(self.files_h)-1) # minus 1 cause there is one file unequal h and l
            self.fi_ys = []
            i_ofs = 0
            for i in tqdm(range(len(self.files_h))):
                rf_data_h = load_dronerf_raw_single(self.files_h[i], t_seg)
                rf_data_l = load_dronerf_raw_single(self.files_l[i], t_seg)
        
                if len(rf_data_h[-1])!=len(rf_data_l[-1]) or len(rf_data_h)!=len(rf_data_l):
                    print('diff', i, 'file name:', self.files_l[i]) 
                    # not sure why one pair of files have different lengths (skip this file for now)
                    i_ofs = 1
                    i_bad = i
                else:
                    i_eff = i - i_ofs
                    if i ==0:
                        self.fi_lens[i_eff] = len(rf_data_h)
                    else:
                        self.fi_lens[i_eff] = self.fi_lens[i_eff-1]+len(rf_data_h)
                
                    # get label
                    label_name = self.get_label_rf(self.files_h[i], self.output_feat)
                    self.fi_ys.append(label_name)
            # remove file the file list
            self.files_h.pop(i_bad)
            self.files_l.pop(i_bad)

        elif whichdata == 'dronedetect':
            self.files = []
            for sf in tqdm(interferences, desc='get files'):
                drone_folders = os.listdir(main_folder+sf+'/')
                for df in drone_folders:
                    sd_dronefolder = main_folder+sf+'/'+df+'/'
                    ful_file_names = [sd_dronefolder+fi for fi in os.listdir(sd_dronefolder)]
                    self.files.extend(ful_file_names)
#                     print(self.files)
            
            self.fi_lens = np.zeros(len(self.files))
            self.fi_ys = []
            # get the number of splits for one file
            DATA, _ = load_dronedetect_raw(self.files[0], t_seg)
            n_splits = len(DATA)
            for i in tqdm(range(len(self.files)), desc='get file lengths'):
                f = self.files[i]
                # load for each one taking a while
                start = time.time()
                DATA, _ = load_dronedetect_raw(f, t_seg)
                n_splits = len(DATA)
                print('time for loading:', time.time()-start)
                if i ==0:
                    self.fi_lens[i] = n_splits
                else:
                    self.fi_lens[i] = self.fi_lens[i-1]+n_splits
                    
                # get label
                label_name = self.get_label_detect(f, self.output_feat)
                self.fi_ys.append(label_name)
                
        self.fi_lens = self.fi_lens.astype(int)
        self.unique_labels = set(self.fi_ys)
            
        self.class_to_idx = {lbl:i for i,lbl in enumerate(self.unique_labels)}
        self.idx_to_class = {i:lbl for i,lbl in enumerate(self.unique_labels)}
            
    def __len__(self):
        return self.fi_lens[-1]
    
    def __getitem__(self, i):
        if self.whichdata == 'dronerf':
            i_file = np.argwhere(self.fi_lens>i)[0][0]
            rf_data_h = load_dronerf_raw_single(self.files_h[i_file], self.t_seg)
            rf_data_l = load_dronerf_raw_single(self.files_l[i_file], self.t_seg)
            
            i_infile = int(len(rf_data_h)- (self.fi_lens[i_file]-i))
            try:
                Xhigh = torch.tensor(rf_data_h[i_infile]).float()
                Xlow = torch.tensor(rf_data_l[i_infile]).float()
            except:
                print('i_infile is', i_infile)
                print('i is', i)
            if self.downsamp:
                Xhigh = self.down_interp(Xhigh)
                Xlow = self.down_interp(Xlow)
                
            X = torch.stack((Xhigh,Xlow), dim=0)
            y = self.class_to_idx[self.fi_ys[i_file]]
            return X, y
#             return torch.tensor(self.X[i]).float(), torch.tensor(self.y[i]).float()
        elif self.whichdata == 'dronedetect':
            i_file = np.argwhere(self.fi_lens>i)[0][0]
            d_splits, _ = load_dronedetect_raw(self.files[i_file], self.t_seg)
            i_infile = int(len(d_splits)- (self.fi_lens[i_file]-i))
            Xreal = torch.tensor(d_splits[i_infile].real).float()
            Ximag = torch.tensor(d_splits[i_infile].imag).float()
            if self.downsamp:
                Xreal = self.down_interp(Xreal)
                Ximag = self.down_interp(Ximag)
                
            X = torch.stack((Xreal,Ximag), dim=0)
            y = self.class_to_idx[self.fi_ys[i_file]]
            return X, y
    
    def get_label_detect(self, f, output_feat):
#         print('split name final', split_name_final)
        if output_feat == 'drones':
            split_name_final = f.split('/')[-1]
            return split_name_final.split('_')[0]
        elif output_feat == 'modes':
            split_name_final = f.split('/')[-2]
            return split_name_final.split('_')[-1]
        elif output_feat == 'int':
            return f.split('/')[-3]
        
    def get_label_rf(self, f, output_feat):
        f_last = f.split('/')[-1]
        if output_feat == 'bi':
            return f_last[0]
        if output_feat == 'drones':
            if int(f_last[:3]) == 100:
                return 'BEB'
            elif int(f_last[:3]) == 101:
                return 'AR'
            elif int(f_last[:3]) == 110:
                return 'PHA'
            else:
                return 'None'
        
        
    def down_interp(self, x):
        t = np.arange(0, len(x))
        f = interpolate.interp1d(t,x)
        tt = np.linspace(0, len(x)-1, num=self.downsamp)
        return torch.tensor(f(tt)).float()
          
        
        
        
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
        X = torch.tensor((self.Xarr[index])).float()
#         X = X.unsqueeze(0)
        y = torch.tensor((self.yarr[index]))
        return (X, y)

## Loop through all files in a folder
# def loop_files(main_folder):
#     for sf in inteference_folders: # options: ['WIFI', 'BLUE', 'BOTH', 'CLEAN']
#         print('CURRENT FOLDER: ', sf)

#         drone_folders = os.listdir(main_folder+sf+'/')
#         for df in drone_folders:
    
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
