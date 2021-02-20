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

def get_spots(image, threshold_value=2000, pix_apart=8):
    '''
    Return a list containing x,y coordinates of every spot in the image. Spots are defined by being above threshold_value counts and separated by other spots by pix_apart pixels.
    
    Inputs:
    -------
    image [2-D numpy array]
    threshold_value [default:2000]
    pix_apart [default:8]
    
    Outputs:
    --------
    spots (list)
    --> can be unpacked into x and y arrays with y,x = zip(*spots)
    '''
    pix_above_thr = []
    where = np.where(image > threshold_value)
    npix = len(where[0])
    spots = []
    for j in range(int(npix)):

        pix_above_thr.append(np.array([where[0][j], where[1][j]]))
        if j == 0:
            spots.append(pix_above_thr[-1])
        if j > 1:
            d = pix_above_thr[-1] - pix_above_thr[-2]
            dist = np.linalg.norm(d)
            if dist > pix_apart: # new spot!
                spots.append(np.array(pix_above_thr[-1]))            
    return spots