import os 
import numpy as np
from tqdm import tqdm

## Load numerical feature files
# Inputs:
    # - feat_folder: where the files are located
    # - feat_name: choose from 'PSD', 'SPEC'
    # - interferences: choose from ['BLUE', 'WIFI', 'BOTH', 'CLEAN']
    # - nperseg: n per segment, part of the file names
    # - datestr: date feature files were generated
    
def load_features_arr(feat_folder, feat_name, datestr, n_per_seg, interferences):
    sub_folder_name = 'ARR_'+feat_name+'_'+str(n_per_seg)+'_'+datestr+'/'
    
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