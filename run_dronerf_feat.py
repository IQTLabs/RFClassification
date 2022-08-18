### script version of Generate DroneRF Features Notebook 

import os
import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import os
from sklearn.model_selection import train_test_split
from spafe.features.lfcc import lfcc
import spafe.utils.vis as vis
from scipy.signal import get_window
import scipy.fftpack as fft
from scipy import signal
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm

from loading_functions import *
from file_paths import *
from feat_gen_functions import *

import importlib

# Dataset Info
main_folder = dronerf_raw_path
t_seg = 20
Xs_arr, ys_arr, y4s_arr, y10s_arr = load_dronerf_raw(main_folder, t_seg)
fs = 40e6 #40 MHz

print('length of X:', len(Xs_arr), 'length of y:', len(ys_arr))

n_per_seg = 1024 # length of each segment (powers of 2)
n_overlap_spec = 120
win_type = 'hamming' # make ends of each segment match
high_low = 'L' #'L', 'H' # high or low range of frequency
feature_to_save = ['PSD'] # what features to generate and save: SPEC or PSD
format_to_save = ['IMG'] # IMG or ARR or RAW
to_add = True
spec_han_window = np.hanning(n_per_seg)

# Image properties
dim_px = (224, 224) # dimension of image pixels
dpi = 100

# Raw input len
v_samp_len = 10000

# data saving folders
features_folder = dronerf_feat_path
date_string = date.today()
# folder naming: ARR_FEAT_NFFT_SAMPLELENGTH
arr_spec_folder = "ARR_SPEC_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
arr_psd_folder = "ARR_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
img_spec_folder = "IMG_SPEC_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
img_psd_folder = "IMG_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
raw_folder = 'RAW_VOLT_'+str(v_samp_len)+"_"+str(t_seg)+"/" # high and low frequency stacked together

existing_folders = os.listdir(features_folder)

if high_low == 'H':
    i_hl = 0
elif high_low == 'L':
    i_hl = 1
    
# check if this set of parameters already exists
# check if each of the 4 folders exist
sa_save = False   #spec array
si_save = False   #spec imag
pa_save = False   #psd array
pi_save = False   #psd imag
raw_save = False # raw high low signals

if 'SPEC' in feature_to_save:
    if 'ARR' in format_to_save:
        if arr_spec_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+arr_spec_folder)
            except:
                print('folder already exist - adding')
            sa_save = True
            print('Generating SPEC in ARRAY format')
        else:
            print('Spec Arr folder already exists')
    if 'IMG' in format_to_save:
        if img_spec_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+img_spec_folder)
            except:
                print('folder already exist - adding')
            si_save = True
            print('Generating SPEC in IMAGE format')
        else:
            print('Spec Arr folder already exists')
if 'PSD' in feature_to_save:
    if 'ARR' in format_to_save:
        if arr_psd_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+arr_psd_folder)
            except:
                print('folder already exist - adding')
            pa_save = True
            print('Generating PSD in ARRAY format')
        else:
            print('PSD Arr folder already exists')
    if 'IMG' in format_to_save:
        if img_psd_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+img_psd_folder)
            except:
                print('folder already exist - adding')
            pi_save = True
            print('Generating PSD in IMAGE format')
        else:
            print('PSD Arr folder already exists')

if 'RAW' in feature_to_save:
    if raw_folder in existing_folders or to_add:
        try:
            os.mkdir(features_folder+raw_folder)
        except:
            print('RAW V folder already exists')
        raw_save = True


if all([not sa_save, not si_save, not pa_save, not pi_save, not raw_save]):
    print('Features Already Exist - Do Not Generate')
else:
    n_parts = 24 # process the data in 10 parts so memory doesn't overwhelm

    indices_ranges = np.split(np.array(range(len(Xs_arr))), n_parts) 
    for i in range(n_parts):
        BILABEL = []
        DRONELABEL = []
        MODELALBEL = []
        F_PSD = []
        F_SPEC = []
        F_V = []
        ir = indices_ranges[i]
        for j in tqdm(range(len(ir))):
            d_real = Xs_arr[ir[j]][i_hl]
            
            # if save raw data
            if raw_save:
                t = np.arange(0, len(d_real))
                f_high = interpolate.interp1d(t, Xs_arr[ir[j]][0])
                f_low = interpolate.interp1d(t, Xs_arr[ir[j]][1])
                tt = np.linspace(0, len(d_real)-1, num=v_samp_len)

                d_v = np.stack((f_high(tt), f_low(tt)), axis=0)
                F_V.append(d_v)
            
            if pa_save or pi_save:
            # calculate PSD
                fpsd, Pxx_den = signal.welch(d_real, fs, window=win_type, nperseg=n_per_seg)
                if pa_save:
                    F_PSD.append(Pxx_den)
                if pi_save:
                    save_psd_image_rf(features_folder, img_psd_folder,
                                      y10s_arr[ir[j]], i, j, Pxx_den, dim_px, dpi)
            
            if sa_save or si_save:
            # calculate spectrogram
            # welch's method older
#           fspec, t, Sxx = signal.spectrogram(d_real, fs, window=win_type, nperseg=n_per_seg)
            
                if si_save: # set up fig properties if saving images
                    plt.clf()
                    fig,ax = plt.subplots(1, figsize=(dim_px[0]/dpi, dim_px[1]/dpi), dpi=dpi)
                    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
                    ax.axis('tight')
                    ax.axis('off')

                spec, _, _, _ = plt.specgram(d_real, NFFT=n_per_seg, Fs=fs, window=spec_han_window, 
                                  noverlap=n_overlap_spec, sides='onesided')
                if si_save:
                    save_spec_image_fig_rf(features_folder, img_spec_folder, 
                                           y10s_arr[ir[j]], i, j, fig, dpi)
                if sa_save:
                    F_SPEC.append(interpolate_2d(Sxx, (224,224)))

            # Labels
            BILABEL.append(ys_arr[ir[j]])
            DRONELABEL.append(y4s_arr[ir[j]])
            MODELALBEL.append(y10s_arr[ir[j]])
        
        if sa_save:
            save_array_rf(features_folder+arr_spec_folder, F_SPEC, BILABEL, DRONELABEL, MODELALBEL, 'SPEC', n_per_seg, i)
        if pa_save:
            save_array_rf(features_folder+arr_psd_folder, F_PSD, BILABEL, DRONELABEL, MODELALBEL, 'PSD', n_per_seg, i)
        if raw_save:
            save_array_rf(features_folder+raw_folder, F_V, BILABEL, DRONELABEL, MODELALBEL, 'RAW', '', i)