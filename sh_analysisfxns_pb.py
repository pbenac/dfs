import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom, rotate
import glob
import matplotlib.pyplot as plt
from shanalyzefull import *

def raw_to_npy(filelist, rawsize=262150, newsize=(512,512), idxcutoff=6):
    '''
    Converts a filelist of .raw files (e.g. the output from the Advanced frame viewer on cred-sao) to npy arrays
    
    Inputs:
    ---
    filelist (list of str): a list of filenames (I use absolute paths) 
    rawsize (int): the size of the raw files. (default = 262150, the picam value)
    newsize (tuple of int): the size of the reshaped npy arrays (default = (512,512))
    idxcutoff (int): index data begins at (default  = 6, picam value)
    
    --> relation between sizes: rawsize = (newsize[0]*newsize[1]) + idxcutoff
    
    Outputs:
    ---
    a list with two entries:
    [all_data, all_images]
    all_data has shape (len(filelist), rawsize) and contains the unreshaped data as npy arrays
    all_images has shape (len(filelist), newsize[0], newsize[1]) and contains the reshaped data 
    --> e.g. plt.imshow(all_images[0])
    '''
    all_data = np.empty(shape=(len(filelist),rawsize))
    all_images = np.empty(shape=(len(filelist),newsize[0],newsize[1]))
    for i, file in enumerate(filelist):
        all_data[i] = np.fromfile(file, dtype=np.short)
        all_images[i] = all_data[i][idxcutoff:].reshape(newsize)  
    return [all_data, all_images]