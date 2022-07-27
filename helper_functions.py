import os 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib

# function to load array files as images
# data in numpy array 1 or 2D
# inputs: data, dimensions in pixels, and dpi
def plot_feat(data, dim_pix, dpi, to_show=True, show_axis=True): 
    fig = plt.figure(figsize=(dim_pix[0]/dpi, dim_pix[1]/dpi), dpi=dpi)
    plt.ioff()
    if len(data.shape) == 2:   # spectrogram - 2D data
        plt.pcolormesh(data, cmap='viridis', vmin=data.min(), vmax=data.max())
    elif len(data.shape) == 1: # psd - 1D data
        plt.plot(data)
        if not show_axis:
            plt.axis('off')
        if to_show:
            plt.show()
        else:
            plt.close(fig)
    return fig
        
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
            r = (np.max(rf[i][j])-np.min(rf[i][j]))
            m = np.min(rf[i][j])
            rfnorm_i[j] = (rf[i][j]-m)/r
        rfnorm.append(rfnorm_i)

    return rfnorm

## convert figure object to 3channel array
from matplotlib.backends.backend_agg import FigureCanvasAgg

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels 
     and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """

    canvas = FigureCanvasAgg(fig)
    # draw the renderer
    fig.canvas.draw ( )
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to 
#      have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


## function to reload package when developing
## packagename in string
# def reload_package(packagename):
# #     importcom = 'import '+packagename
# #     eval(importcom)
#     importlib.import_module(packagename)
#     importlib.reload(packagename)
# #     importlibcom = 'importlib.reload('+packagename+')'
# #     eval(importlibcom)
#     reload_com = 'from '+packagename+ ' import *'
#     eval(reload_com)
# #     from helper_functions import *