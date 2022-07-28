# scripts designed to run on raspberry Pi

import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is

from gamutrf_feature_functions import *
from gamutrf.sample_reader import read_recording
from gamutrf.utils import parse_filename
import pickle
import time
import os
import argparse

import matplotlib.pyplot as plt 
from Torch_Models import *


def run_psd_svm(n_per_seg, t_seg):
#     n_per_seg = 256
#     t_seg = 20
    model_folder = '../../saved_models/'
    model_file = 'SVM_PSD_'+str(n_per_seg)+'_'+str(t_seg)+'_1'
    win_type = 'hamming'

    model = pickle.load(open(model_folder+model_file, 'rb'))

    ## Generate features
    data_folder = '../../sample_data/'

    for fi in os.listdir(data_folder):
        full_file = data_folder+fi
        if fi.endswith('.zst'):    # check if it is a compressed datafile

            freq_center, sample_rate, sample_dtype, sample_len, sample_type, sample_bits = parse_filename(full_file)
            # read sample
            start_ft = time.time()
            samples = read_recording(full_file, sample_rate, sample_dtype, sample_len, t_seg/1e3)

            freqs, psds = get_PSD_from_samples(samples, sample_rate, win_type, n_per_seg)

            end_ft = time.time()
            print('Feature time:', end_ft-start_ft)

            start_pd = time.time()
            pout = model.predict(psds)

            end_pd = time.time()
            print('Prediction time:', end_pd-start_pd)

            # print average time per sample
            n_samps = pout.shape[0]
            avg_time_feat = (end_ft-start_ft)/n_samps
            avg_time_pred = (end_pd-start_pd)/n_samps

            print('average time for Feature Generation: {:.3}ms, Prediction: {:.3}ms'.format(avg_time_feat*1e3, avg_time_pred*1e3))
            
            
def run_spec_vggfc(n_per_seg, t_seg):
    model_folder = '../../saved_models/'
    model_file = 'VGGFC_SPEC_'+str(n_per_seg)+'_'+str(t_seg)

    ## Load Model
    model = torch.load(model_folder+model_file)

    ## Generate features
    data_folder = '../../sample_data/'

    for fi in os.listdir(data_folder):
        full_file = data_folder+fi
        if fi.endswith('.zst'):    # check if it is a compressed datafile

            freq_center, sample_rate, sample_dtype, sample_len, sample_type, sample_bits = parse_filename(full_file)
            # read sample

            samples = read_recording(full_file, sample_rate, sample_dtype, sample_len, t_seg/1e3)

            # get feature
            start_ft = time.time()
            return_array = True
            rgbs = get_specs_from_samples(samples, sample_rate, n_per_seg, return_array)

            end_ft = time.time()
            print('Feature time:', end_ft-start_ft)
#             print(rgbs.shape)
            n_samps = rgbs.shape[0]
            batchsize = 16

            start_pd = time.time()
            feat = torch.tensor(rgbs/255).float()
            i = 0
            while i+batchsize<n_samps:
                pout = model(feat[i:i+batchsize,:,:])
                i = i+batchsize

            end_pd = time.time()
            print('Prediction time:', end_pd-start_pd)

            # print average time per sample
            
            avg_time_feat = (end_ft-start_ft)/n_samps
            avg_time_pred = (end_pd-start_pd)/n_samps

            print('average time for Feature Generation: {:.3}ms, Prediction: {:.3}ms'.format(avg_time_feat*1e3, avg_time_pred*1e3))
            
            
if __name__ == "__main__":
    ## run from command line: python run_model.py --model spec_vggfc 1024 20

    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--model', required=True)
#     parser.add_argument('model', type=str,
#                         help='enter which model')
    
    parser.add_argument('nperseg', type=int,
                        help='enter fft length')
    parser.add_argument('tseg', type=int,
                    help='enter tims segment')
    args = parser.parse_args()
    
    print("Run Model for:")
    print(args.model+' model')
    print('nFFT = '+ str(args.nperseg))
    print('time segment = '+ str(args.tseg)+'ms')
    if args.model == 'svm_psd':
        run_psd_svm(args.nperseg, args.tseg)
    elif args.model == 'spec_vggfc':
        run_spec_vggfc(args.nperseg, args.tseg)