''' Feature generation and saving functions '''

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from helper_functions import *

## DRONEDETECT - DATA SAVING FUNCTIONS
def interpolate_2d(Sxx_in, output_size):
    x = np.linspace(0, 1, Sxx_in.shape[0])
    y = np.linspace(0, 1, Sxx_in.shape[1])
    f = interpolate.interp2d(y, x, Sxx_in, kind='linear')
    
    x2 = np.linspace(0, 1, output_size[0])
    y2 = np.linspace(0, 1, output_size[1])
    arr2 = f(y2, x2)
    
    return arr2

# save function to save image to file
# def save_spec_image(folder_path, cond_folder, DRONE, COND, INT, FIn, counter, f, t, Sxx):
#     plt.clf()
#     plt.pcolormesh(t, f, Sxx, cmap='Greys', vmin=Sxx.min(), vmax=Sxx.max())
#     full_img_path = folder_path+cond_folder+DRONE+'_'+COND+'_'+INT+'_'+FIn+'_'+str(counter)+'.jpg'
#     plt.savefig(full_img_path)

# save spectrogram when passing in fig object
def save_spec_image_fig(folder_path, cond_folder, DRONE, COND, INT, FIn, counter, fig, dpi):
    full_img_path = folder_path+"/"+cond_folder+"/"+DRONE+'_'+COND+'_'+INT+'_'+FIn+'_'+str(counter)+'.jpg'
    fig.savefig(full_img_path, dpi=dpi)
    plt.close(fig)
    plt.clf()
    
def save_psd_image(folder_path, cond_folder, DRONE, COND, INT, FIn, counter, PSD, dim_px, dpi):
#     plt.clf()
#     plt.plot(f, PSD, 'k')
    fig = plot_feat(PSD, dim_px, dpi, to_show=False, show_axis=False)
    full_img_path = folder_path+"/"+cond_folder+"/"+DRONE+'_'+COND+'_'+INT+'_'+FIn+'_'+str(counter)+'.jpg'
    fig.savefig(full_img_path)    

def save_array_detect(folder_path, feat, DRONES, CONDS, INTS, feat_name, int_name, n_per_seg):
    Xs_arr = np.array(feat)
    
    # labels
    y_drones_arr = np.array(DRONES)
    y_conds_arr = np.array(CONDS)
    y_ints_arr = np.array(INTS)

    data_save = {'feat': Xs_arr, 'drones': y_drones_arr, 'conds': y_conds_arr, 'ints': y_ints_arr}

#    #Save data
#     date_string = date.today()
    fp = folder_path+"/"+int_name+"_"+feat_name+"_"+str(n_per_seg)
    print(fp)
    np.save(fp, data_save)
    
    
### Drone Detect Custom functions

def save_array_rf(folder_path, feat, BI, DRONES, MODES, feat_name, seg_i):
    Xs_arr = np.array(feat)
    
    # labels
    y_bi_arr = np.array(BI)
    y_drones_arr = np.array(DRONES)
    y_modes_arr = np.array(MODES)

    data_save = {'feat': Xs_arr, 'bi': y_bi_arr, 'drones': y_drones_arr, 'modes': y_modes_arr}
#     return data_save

#     #Save data
    date_string = date.today()
    fp = folder_path+feat_name+"_"+str(n_per_seg)+"_"+str(seg_i)
    print(fp)
    np.save(fp, data_save)
    
    
def save_psd_image_rf(folder_path, cond_folder, BUI, group_counter, count, PSD, dim_px, dpi):
    fig = plot_feat(PSD, dim_px, dpi, to_show=False, show_axis=False)
    bui_string = str(BUI).zfill(5)
    full_img_path = folder_path+"/"+cond_folder+"/"+bui_string+'_'+str(group_counter)+'_'+str(count)+'.jpg'
    fig.savefig(full_img_path)
    
def save_spec_image_fig_rf(folder_path, cond_folder, BUI, group_counter, count, fig, dpi):
    bui_string = str(BUI).zfill(5)
    full_img_path = folder_path+"/"+cond_folder+"/"+bui_string+'_'+str(group_counter)+'_'+str(count)+'.jpg'
    fig.savefig(full_img_path, dpi=dpi)
    plt.close(fig)
    plt.clf()
    