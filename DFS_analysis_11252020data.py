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

def rotate_and_ft(images, rotation_angle):
    '''
    inputs:
    images: list of images (2d arrays)
    rotation angle [DEGREES]: angle to rotate by
    
    outputs:
    [rotated_images, FT_rotated_images]
    '''
    rotated_images = []
    FT_rotated_images = []
    for i, image in enumerate(images):
        rotated_images.append(rotate(image, rotation_angle))
        FT_rotated_images.append(np.fft.fftshift(np.fft.fft2(rotate(image, rotation_angle))))
    return [rotated_images, FT_rotated_images]

# I am going to analyze both batches 3 and 6 and use the fringes from batch 5 as the reference
datadir = '/home/pbenac/home/Thesis/11252020_fringedata'

fr_filelist = glob.glob(datadir + '/*')
fr_filelist.sort()

batch3_filelist = glob.glob(datadir + "/*3-*.fits")
batch6_filelist = glob.glob(datadir + "/*6-*.fits")
batch5_filelist = glob.glob(datadir + "/*5-*.fits")

# Load in Appropriate dark

hdul = fits.open('/home/pbenac/home/Thesis/ninox_images/11062020_dark_1100msexpo')
dark_1100ms = hdul[0].data
dark_1100ms = dark_1100ms[0][:][:]


batch6_data = open_fits_filelist(batch6_filelist)
batch3_data = open_fits_filelist(batch3_filelist)
batch5_data = open_fits_filelist(batch5_filelist)

# Dark Correction 
dc_batch6 = []
dc_batch3 = []
dc_batch5 = []

for file in batch6_data:
    dc_batch6.append(file - dark_1100ms)
    
for file in batch3_data:
    dc_batch3.append(file - dark_1100ms)
    
for file in batch5_data:
    dc_batch5.append(file - dark_1100ms)
[batch6_rotated, FT_batch6_rotated] = rotate_and_ft(dc_batch6, -53)
[batch3_rotated, FT_batch3_rotated] = rotate_and_ft(dc_batch3, -53)
[batch5_rotated, FT_batch5_rotated] = rotate_and_ft(dc_batch5, -53)  

# center of box on fringes
rot_xcen = [410]
rot_ycen=[260]

pixsize = 0.07 * (16/24)
d = DFS(pixsize=pixsize)

wavefrontmap = np.zeros((80,80))
wavefrontmap[:,40:] = 1

boxsize = 40
imsize = 100
edgewidth = 8
# ccmin = int(53 * boxsize / 80)
# ccmin = int(50 * boxsize / 80)

# ccmax = int(77 * boxsize / 80)
# ccmax = int(52 *boxsize/80)
ccmin = int(77 * boxsize / 80) #77
ccmax = int(60 * boxsize / 80) #60

dispersion = 5 # arcsec per micron
PixelsPerMicron = dispersion / pixsize
Wavelength = ( 1.358 + 1.027 ) / 2.

# Conversion of pixel shift of FFT peak to nm of piston error
PixPerNm = boxsize  / PixelsPerMicron / Wavelength**2 / 1000.

pd_ref = PhaseData(batch5_rotated[1], xcen=rot_xcen, ycen=rot_ycen, n=50, edgewidth=5)
pd_6_1 = PhaseData(batch6_rotated[0], xcen=rot_xcen, ycen=rot_ycen, n=50, edgewidth=5)
pd_6_2 = PhaseData(batch6_rotated[1], xcen=rot_xcen, ycen=rot_ycen, n=50, edgewidth=5)
pd_6_3 = PhaseData(batch6_rotated[2], xcen=rot_xcen, ycen=rot_ycen, n=50, edgewidth=5)
pd_6_4 = PhaseData(batch6_rotated[3], xcen=rot_xcen, ycen=rot_ycen, n=50, edgewidth=5)
pd_6_5 = PhaseData(batch6_rotated[4], xcen=rot_xcen, ycen=rot_ycen, n=50, edgewidth=5)
pd_6_6 = PhaseData(batch6_rotated[5], xcen=rot_xcen, ycen=rot_ycen, n=50, edgewidth=5)

pd_6_1.computeresults(pd_ref, ccmin, ccmax, PixPerNm, Wavelength)
pd_6_2.computeresults(pd_ref, ccmin, ccmax, PixPerNm, Wavelength)
pd_6_3.computeresults(pd_ref, ccmin, ccmax, PixPerNm, Wavelength)
pd_6_4.computeresults(pd_ref, ccmin, ccmax, PixPerNm, Wavelength)
pd_6_5.computeresults(pd_ref, ccmin, ccmax, PixPerNm, Wavelength)
pd_6_6.computeresults(pd_ref, ccmin, ccmax, PixPerNm, Wavelength)

print('Displacement of img 1', pd_6_1.dfspiston) # in nanometers
print('Displacement of img 2', pd_6_2.dfspiston) # in nanometers
print('Displacement of img 3', pd_6_3.dfspiston) # in nanometers
print('Displacement of img 4', pd_6_4.dfspiston) # in nanometers
print('Displacement of img 5', pd_6_5.dfspiston) # in nanometers
print('Displacement of img 6', pd_6_6.dfspiston) # in nanometers



    