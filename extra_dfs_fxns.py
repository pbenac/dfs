
# IMPORTS
from dfssim import DFS
import numpy as np
from  phasesensor import *

from astropy.io import fits
from scipy.ndimage import zoom, rotate
import glob
import matplotlib.pyplot as plt

def open_fits_filelist(filelist):
    '''
    Open a list of filenames (.fits) as 2d arrays (images)
    Input:
    Filelist - 
    a list of absolute path filenames to fits files, as strings.
    
    Output:
    data - 
    a list of 2D arrays representing those images
    '''
    data = []
    for file in filelist:
        hdul = fits.open(file)
        data.append(hdul[0].data)
        
    for i, image in enumerate(data):
        data[i] = image[0][:][:]
        
    return data

def rotate_and_ft(images, rotation_angle = -53):
    '''
    inputs:
    images: list of images (2d arrays)
    rotation angle [DEGREES]: angle to rotate by (default -53)
    
    outputs:
    [rotated_images, FT_rotated_images]
    '''
    rotated_images = []
    FT_rotated_images = []
    for i, image in enumerate(images):
        rotated_images.append(rotate(image, rotation_angle))
        FT_rotated_images.append(np.fft.fftshift(np.fft.fft2(rotated_images[i])))
    return [rotated_images, FT_rotated_images]