import os 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# function to load array files as images
# data in numpy array 1 or 2D
def plot_image(data, dim, dpi):
    fig = plt.figure(figsize=dim, dpi=dpi)
    if len(data.shape) == 2:   # spectrogram - 2D data
        plt.pcolormesh(data, cmap='Greys', vmin=data.min(), vmax=data.max())
    elif len(data.shape) == 1: # psd - 1D data
        plt.plot(data)
        plt.show()
        
# function to get the number of parameters in the model        
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


## method to normalize in rfuav-net
def normalize_rf(rf):
    """apply normalization to data in the numpy array format"""
    rfnorm = []
    for i in range(len(rf)):
        rfnorm_i = np.zeros(rf[i].shape)
        for j in range(2):
#             print(rf[i][j])
            r = (np.max(rf[i][j])-np.min(rf[i][j]))
#             print('range:', r)
            m = np.min(rf[i][j])
#             print('m:', m)abs
            rfnorm_i[j] = (rf[i][j]-m)/r
#             print(rfnorm_i[j])
        rfnorm.append(rfnorm_i)

    return rfnorm