'''
Compute & save prediction features from gamutRF collected samples
'''

from scipy import signal
import numpy as np

def get_PSD_from_samples(samples, fs, win_type, n_per_seg):
    psd_return = []
    for i, s in enumerate(samples):
        fpsd, Pxx_den = signal.welch(s, fs, window=win_type, nperseg=n_per_seg)
        psd_return.append(Pxx_den)
    return fpsd, np.array(psd_return) # frequency range should be the same for each sample

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

def save_psd_img(psd_array, dim_px, dpi, full_file):
#     def save_psd_image(folder_path, cond_folder, DRONE, COND, INT, FIn, counter, PSD, dim_px, dpi):
#     plt.clf()
#     plt.plot(f, PSD, 'k')
    folder_name = 'Features/PSD_IMG/'
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