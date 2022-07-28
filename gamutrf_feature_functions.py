'''
Compute & save prediction features from gamutRF collected samples
'''

from scipy import signal
import numpy as np
from helper_functions import *

# return in arrays
def get_PSD_arr_from_samples(samples, fs, win_type, n_per_seg):
#     psd_return = []
    for i, s in enumerate(samples):
        fpsd, Pxx_den = signal.welch(s, fs, window=win_type, nperseg=n_per_seg, return_onesided=False)
        # return one sided is false given input data is complex
#         Pxx_den = Pxx_den.reshape((1,len(Pxx_den)))
        try:
            if len(Pxx_den)==n_per_seg:
                psd_return = np.vstack((psd_return, Pxx_den))
        except:
            psd_return = Pxx_den
    return fpsd, psd_return # frequency range should be the same for each sample

# generate spectrogram from samples generator object
# if return spec in rgb arrays, use return_array= True, otherwise return a figure list
def get_specs_img_from_samples(samples, fs, n_per_seg, return_array, dim_px=(224,224), dpi=100, noverlap=120):
    if return_array:
        rbg_ls = []
    else:
        fig_ls = []
    for i, sa in enumerate(samples):
        plt.close()
        fig,ax = plt.subplots(1, figsize=(dim_px[0]/dpi, dim_px[1]/dpi), dpi=dpi)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.axis('tight')
        ax.axis('off')
        spec, _, _, _ = plt.specgram(sa, NFFT=n_per_seg, Fs=fs,
                                     noverlap=120, sides='onesided') # use default window
        if return_array:
            rgba = fig2data(fig)
            rgb = rgba[:,:,:3]
            rbg_ls.append(rgb)
        else:
            fig_ls.append(fig)
            
    if return_array:
        rbg_return = np.stack(rbg_ls, axis=0)
        return rbg_return
    else:
        return fig_ls

# generate psd images from psd arrays
def get_psd_img(psd_array, return_array, dim_px=(224,224), dpi=100):
    if return_array:
        rbg_ls = []
    else:
        fig_ls = []
    for i in range(psd_array.shape[0]):
        fig = plot_feat(psd_array[i,:], dim_px, dpi, to_show=False, show_axis=False)
        if return_array:
            rgba = fig2data(fig)
            rgb = rgba[:,:,:3]
            rbg_ls.append(rgb)
        else:
            fig_ls.append(fig)
            
    if return_array:
        rbg_return = np.stack(rbg_ls, axis=0)
        return rbg_return
    else:
        return fig_ls
    

# saving options
def save_psd_array(freqs, psd_array, full_file):
    data_save = {'feat': psd_array, 'freq': freqs}
    
    # construct file name
    ss = full_file.split('/')
    folder_name = 'Features/PSD_ARR/'
    ss[-1] = folder_name+'/psd_'+ss[-1]
    new_name = '/'.join(ss)
    
    folder_name_full = '/'.join(ss[:-1])
    folder_name_full = folder_name_full+'/'+folder_name

    #     #Save data
    print(new_name)
    try:
        np.save(new_name, data_save)
    except:
        folder_name_full = '/'.join(ss[:-1])
        folder_name_full = folder_name_full+'/'+folder_name
        os.mkdir(folder_name_full)
        np.save(new_name, data_save)

def save_psd_img(psd_array, dim_px, dpi, full_file, folder_name):
#     def save_psd_image(folder_path, cond_folder, DRONE, COND, INT, FIn, counter, PSD, dim_px, dpi):
#     plt.clf()
#     plt.plot(f, PSD, 'k')
#     folder_name = 'Features/PSD_IMG_'/'
    ss = full_file.split('/')
    ss[-1] = folder_name+'/psd_'+ss[-1]
    new_name =  '/'.join(ss)
    
    print(new_name)
    
    for i, PSD in enumerate(psd_array):
        fig = plot_feat(PSD, dim_px, dpi, to_show=False, show_axis=False)
        fig_name = new_name+'_'+str(i)+'.jpg'
        fig.savefig(fig_name)

def save_spec_array(Z, extent, full_file):
    data_save = {'feat': Z, 'freq': extent[2:], 'time': extent[:2]}
    
    # construct file name
    date_string = date.today()
    ss = full_file.split('/')
    folder_name = 'Features/SPEC_'+str(date_string)
    ss[-1] = folder_name+'/spec_'+ss[-1]
    new_name = '/'.join(ss)
    
#     #Save data
    print(new_name)
    try:
        np.save(new_name, data_save)
    except:
        folder_name_full = '/'.join(ss[:-1])
        folder_name_full = folder_name_full+'/'+folder_name
        os.mkdir(folder_name_full)
        np.save(new_name, data_save)