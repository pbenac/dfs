#!/usr/bin/python3
import astropy.io.fits as fits
import numpy as np
import pylab as pl
from scipy.signal import correlate, correlate2d
from scipy.ndimage.measurements import center_of_mass
import scipy.special
from scipy import ndimage
import os
from argparse import ArgumentParser

from phasesensor import *

import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def get_fringe_config(fits_hdr,box_num):
    # Pass in a FITS header from a AGWS processed image, and this returns
    # the parameters that describe the fringe pattern contained in that box.
    # fits_hdr: pyfits header from primary header of AGWS processed image
    # box_num: 1 to 3
    # returns: (box_xpos,box_ypos,box_oadist)

    #El 83 zenith case
    #upper_box1_xcenter = 194.7   #Updated
    #upper_box1_ycenter = 202.1

    #upper_box2_xcenter = 225.5   #Updated
    #upper_box2_ycenter = 151.8

    #upper_box3_xcenter = 165.5   #Updated
    #upper_box3_ycenter = 151.3

    
    #el 60 stars
    #upper_box1_xcenter = 203.7   #Updated
    #upper_box1_ycenter = 207.1

    #upper_box2_xcenter = 236.5   #Updated
    #upper_box2_ycenter = 151.8

    #upper_box3_xcenter = 173   #Updated
    #upper_box3_ycenter = 151.3

    #Rising in east
    #upper_box1_xcenter = 192.7   #Updated
    #upper_box1_ycenter = 199.1

    #upper_box2_xcenter = 227.5   #Updated
    #upper_box2_ycenter = 143.8

    #upper_box3_xcenter = 162   #Updated
    #upper_box3_ycenter = 144

    #Star Shifts
    #upper_box1_xcenter = 200.7   #Updated
    #upper_box1_ycenter = 193.1

    #upper_box2_xcenter = 233.5   #Updated
    #upper_box2_ycenter = 146.8

    #upper_box3_xcenter = 168   #Updated
    #upper_box3_ycenter = 146

    #Star in transit
    upper_box1_xcenter = 186.7   #Updated
    upper_box1_ycenter = 198.1

    upper_box2_xcenter = 220.5   #Updated
    upper_box2_ycenter = 146.8

    upper_box3_xcenter = 157   #Updated
    upper_box3_ycenter = 146
    
    
    
    upper180_box1_xcenter = 210.5
    upper180_box1_ycenter = 198.75

    upper180_box2_xcenter = 243
    upper180_box2_ycenter = 144.75

    upper180_box3_xcenter = 180
    upper180_box3_ycenter = 142.25


    lower_box1_xcenter = 205
    lower_box1_ycenter = 116.6

    lower_box2_xcenter = 175.25
    lower_box2_ycenter = 167.8

    lower_box3_xcenter = 236.75
    lower_box3_ycenter = 170.1

    lower180_box1_xcenter = 205.5
    lower180_box1_ycenter = 112.1

    lower180_box2_xcenter = 173.6
    lower180_box2_ycenter = 165.8

    lower180_box3_xcenter = 237
    lower180_box3_ycenter = 166

    if 'MASKNAME' in hdr.keys():
        if (fits_hdr['MASKNAME'] == 'Mask A' or fits_hdr['MASKNAME'] == 'maska'):
            if fits_hdr['PRSMMODE'] == 'prism_upper':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(upper_box1_xcenter,upper_box1_ycenter,8)
                    elif box_num == 2:
                        return(upper_box2_xcenter,upper_box2_ycenter,6)
                    elif box_num == 3:
                        return(upper_box3_xcenter,upper_box3_ycenter,0)
                elif fits_hdr['PRSMANG'] == 180:
                    if box_num == 1:
                        return(upper180_box1_xcenter,lower_box1_ycenter,6)
                    elif box_num == 2:
                        return(upper180_box2_xcenter,lower_box2_ycenter,8)
                    elif box_num == 3:
                        return(upper180_box3_xcenter,lower_box3_ycenter,0)
            elif fits_hdr['PRSMMODE'] == 'prism_lower':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(lower_box1_xcenter,lower_box1_ycenter,6)
                    elif box_num == 2:
                        return(lower_box2_xcenter,lower_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower_box3_xcenter,lower_box3_ycenter,6)
                elif fits_hdr['PRSMANG'] == 180:
                    if box_num == 1:
                        return(lower180_box1_xcenter,lower180_box1_ycenter,8)
                    elif box_num == 2:
                        return(lower180_box2_xcenter,lower180_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower180_box3_xcenter,lower180_box3_ycenter,8)
            else:
                print("Prism mode not correct")
                exit
        elif (fits_hdr['MASKNAME'] == 'Mask B' or fits_hdr['MASKNAME'] == 'maskb'):
            if fits_hdr['PRSMMODE'] == 'prism_upper':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(upper_box1_xcenter,upper_box1_ycenter,0)
                    elif box_num == 2:
                        return(upper_box2_xcenter,upper_box2_ycenter,0)
                    elif box_num == 3:
                        return(upper_box3_xcenter,upper_box3_ycenter,0)
                elif fits_hdr['PRSMANG'] == 180:
                    if box_num == 1:
                        return(upper180_box1_xcenter,lower_box1_ycenter,0)
                    elif box_num == 2:
                        return(upper180_box2_xcenter,lower_box2_ycenter,0)
                    elif box_num == 3:
                        return(upper180_box3_xcenter,lower_box3_ycenter,0)
            elif fits_hdr['PRSMMODE'] == 'prism_lower':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(lower_box1_xcenter,lower_box1_ycenter,0)
                    elif box_num == 2:
                        return(lower_box2_xcenter,lower_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower_box3_xcenter,lower_box3_ycenter,0)
                elif fits_hdr['PRISMANG'] == 180:
                    if box_num == 1:
                        return(lower180_box1_xcenter,lower180_box1_ycenter,0)
                    elif box_num == 2:
                        return(lower180_box2_xcenter,lower180_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower180_box3_xcenter,lower180_box3_ycenter,0)
            else:
                print("Prism mode not correct")
                exit
        else:
            print("Unknown mask type")
            exit
    else:
        return float(fits_hdr['NAXIS1'])/2, float(fits_hdr['NAXIS1'])/2, 0

def get_ref_fringe_config(fits_hdr,box_num):
    # Pass in a FITS header from a AGWS processed image, and this returns
    # the parameters that describe the fringe pattern contained in that box.
    # fits_hdr: pyfits header from primary header of AGWS processed image
    # box_num: 1 to 3
    # returns: (box_xpos,box_ypos,box_oadist)

    #El sweep ref
    #upper_box1_xcenter = 179.6   #Updated
    #upper_box1_ycenter = 202.3

    #upper_box2_xcenter = 209.3   #Updated
    #upper_box2_ycenter = 153.8

    #upper_box3_xcenter = 152.7   #Updated
    #upper_box3_ycenter = 153.3

    #Next day rising in east ref
    #upper_box1_xcenter = 180.7   #Updated
    #upper_box1_ycenter = 199.1

    #upper_box2_xcenter = 209.5   #Updated
    #upper_box2_ycenter = 151.8

    #upper_box3_xcenter = 153   #Updated
    #upper_box3_ycenter = 151.3

    # Star shifts
    upper_box1_xcenter = 179.7   #Updated
    upper_box1_ycenter = 201.1

    upper_box2_xcenter = 210.5   #Updated
    upper_box2_ycenter = 151.8

    upper_box3_xcenter = 153   #Updated
    upper_box3_ycenter = 151.3

    
    upper180_box1_xcenter = 204.5
    upper180_box1_ycenter = 187.75

    upper180_box2_xcenter = 237
    upper180_box2_ycenter = 134.75

    upper180_box3_xcenter = 173
    upper180_box3_ycenter = 134.25


    lower_box1_xcenter = 205
    lower_box1_ycenter = 116.6

    lower_box2_xcenter = 175.25
    lower_box2_ycenter = 167.8

    lower_box3_xcenter = 236.75
    lower_box3_ycenter = 170.1

    lower180_box1_xcenter = 205.5
    lower180_box1_ycenter = 112.1

    lower180_box2_xcenter = 173.6
    lower180_box2_ycenter = 165.8

    lower180_box3_xcenter = 237
    lower180_box3_ycenter = 166

    if 'MASKNAME' in fits_hdr.keys():
        if (fits_hdr['MASKNAME'] == 'Mask A' or fits_hdr['MASKNAME'] == 'maska'):
            if fits_hdr['PRSMMODE'] == 'prism_upper':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(upper_box1_xcenter,upper_box1_ycenter,8)
                    elif box_num == 2:
                        return(upper_box2_xcenter,upper_box2_ycenter,6)
                    elif box_num == 3:
                        return(upper_box3_xcenter,upper_box3_ycenter,0)
                elif fits_hdr['PRSMANG'] == 180:
                    if box_num == 1:
                        return(upper180_box1_xcenter,lower_box1_ycenter,6)
                    elif box_num == 2:
                        return(upper180_box2_xcenter,lower_box2_ycenter,8)
                    elif box_num == 3:
                        return(upper180_box3_xcenter,lower_box3_ycenter,0)
            elif fits_hdr['PRSMMODE'] == 'prism_lower':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(lower_box1_xcenter,lower_box1_ycenter,6)
                    elif box_num == 2:
                        return(lower_box2_xcenter,lower_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower_box3_xcenter,lower_box3_ycenter,6)
                elif fits_hdr['PRSMANG'] == 180:
                    if box_num == 1:
                        return(lower180_box1_xcenter,lower180_box1_ycenter,8)
                    elif box_num == 2:
                        return(lower180_box2_xcenter,lower180_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower180_box3_xcenter,lower180_box3_ycenter,8)
            else:
                print("Prism mode not correct")
                exit
        elif (fits_hdr['MASKNAME'] == 'Mask B' or fits_hdr['MASKNAME'] == 'maskb'):
            if fits_hdr['PRSMMODE'] == 'prism_upper':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(upper_box1_xcenter,upper_box1_ycenter,0)
                    elif box_num == 2:
                        return(upper_box2_xcenter,upper_box2_ycenter,0)
                    elif box_num == 3:
                        return(upper_box3_xcenter,upper_box3_ycenter,0)
                elif fits_hdr['PRSMANG'] == 180:
                    if box_num == 1:
                        return(upper180_box1_xcenter,lower_box1_ycenter,0)
                    elif box_num == 2:
                        return(upper180_box2_xcenter,lower_box2_ycenter,0)
                    elif box_num == 3:
                        return(upper180_box3_xcenter,lower_box3_ycenter,0)
            elif fits_hdr['PRSMMODE'] == 'prism_lower':
                if fits_hdr['PRISMANG'] == 0:
                    if box_num == 1:
                        return(lower_box1_xcenter,lower_box1_ycenter,0)
                    elif box_num == 2:
                        return(lower_box2_xcenter,lower_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower_box3_xcenter,lower_box3_ycenter,0)
                elif fits_hdr['PRISMANG'] == 180:
                    if box_num == 1:
                        return(lower180_box1_xcenter,lower180_box1_ycenter,0)
                    elif box_num == 2:
                        return(lower180_box2_xcenter,lower180_box2_ycenter,0)
                    elif box_num == 3:
                        return(lower180_box3_xcenter,lower180_box3_ycenter,0)
            else:
                print("Prism mode not correct")
                exit
        else:
            print("Unknown mask type")
            exit        
    else:
        return float(fits_hdr['NAXIS1'])/2, float(fits_hdr['NAXIS1'])/2, 0


def proto3_centers ( image, fits_hdr ):
    sigma = 16
    template = np.zeros(image.shape)
    iy, ix = np.indices(template.shape)
    
    for box_num in [1, 2, 3]:
        boxx, boxy, oadist = get_fringe_config(fits_hdr, box_num)

        spot = np.exp(-((iy - boxy)**2 + (ix - boxx)**2) / 2 / sigma)
        template = template + spot

    xcor = np.real(np.fft.fftshift(np.fft.fft2(np.conj(np.fft.fft2(image)) * np.fft.fft2( template))))
    shifty, shiftx = np.unravel_index(np.argmax(xcor), xcor.shape)
    shifty = shifty - xcor.shape[0]//2
    shiftx = shiftx - xcor.shape[1]//2
    eprint("Shift", shiftx, shifty)
    return shiftx, shifty


# Default parameters
#
config = None
darkf  = None
flatf   = None
x0     = 151
y0     = 151
selfcalib = False
ceniters  = 0
boxsize   = 40
windowradius = 14
windowfwhm    = 4
refimagef    = "simulatedfringes.fits"
edgewidth     = 8
boxargs = False
refboxargs = False
summarize = False
box1x = 0
box1y = 0
box2x = 0
box2y = 0
box3x = 0
box3y = 0

r0ref = 62
rref = [0,r0ref,r0ref]
aref = np.array([0, 60, 120])
xcenref = (188 + rref * np.cos(np.radians(aref))).astype(int)
ycenref = (101 + rref * np.sin(np.radians(aref))).astype(int)
# The simulated image fringes correspond to the 0', 8', and 6' OA positions
oadistsref = np.array([0,8,6],dtype=np.int32)


# Default box positions in the case that FITS headers don't specify inst config
r0 = 64
r = [0,r0,r0]
a = np.array([0, 0, 60])
xcenimg =  (x0 + r * np.cos(np.radians(a))).astype(int)
ycenimg =  (y0 + r * np.sin(np.radians(a))).astype(int)
oadistsimg = np.array([0,0,0])

# More default values that could be overridden in the config file
#
pinhole_dia_mm = 0.025

# Spectral dispersion
PixelsPerMicron = 16.6 / ( 1.34 - 1.11 )
Wavelength = ( 1.358 + 1.027 ) / 2.

# Conversion of pixel shift of FFT peak to nm of piston error
PixPerNm = boxsize  / PixelsPerMicron / Wavelength**2 / 1000.

# Pixel size
pixsize = 0.072

# Pinhole size in detector pixels
pinhole_rad_pix = pinhole_dia_mm / 2 * 2 / pixsize  # 0.025/2 mm pinhole  radius * 2"/mm ( 0.070"/CREDpix * 15um Ninox/24um CRED)

# Columns of reference image FFT to use as the cross-correlation kernel
ccmin = int(53 * boxsize / 80)
ccmax = int(77 * boxsize / 80)

# Replace defaults with the contents of the config file
#
parser = ArgumentParser()
parser.add_argument("--config",       type=str,       dest="config",       default=None)

(arguments, args) = parser.parse_known_args()
if (arguments.config!=None):
    exec(open(arguments.config,'r').read())

# Finally override anything in the config file with arguments passed on the command line
#
parser.add_argument("--darkf",        type=str,           default=darkf)
parser.add_argument("--flatf",        type=str,           default=flatf)
parser.add_argument("--x0",           type=float,         default=x0)
parser.add_argument("--y0",           type=float,         default=y0)
parser.add_argument("--selfcalib",    action="store_true",default=selfcalib)
parser.add_argument("--ceniters",     type=int,           default=ceniters)
parser.add_argument("--boxsize",      type=int,           default=boxsize)
parser.add_argument("--windowradius", type=float,         default=windowradius)
parser.add_argument("--windowfwhm",   type=float,         default=windowfwhm)
parser.add_argument("--refimagef",    type=str,           default=refimagef)
parser.add_argument("--edgewidth",    type=int,           default=edgewidth)
parser.add_argument("--boxargs",      action="store_true",default=boxargs)
parser.add_argument("--refboxargs",      action="store_true",default=refboxargs)
parser.add_argument("--box1x",        type=float,         default=box1x)
parser.add_argument("--box1y",        type=float,         default=box1y)
parser.add_argument("--box2x",        type=float,         default=box2x)
parser.add_argument("--box2y",        type=float,         default=box2y)
parser.add_argument("--box3x",        type=float,         default=box3x)
parser.add_argument("--box3y",        type=float,         default=box3y)
parser.add_argument("--summarize",    action="store_true", default=summarize)

arguments, args = parser.parse_known_args(args)

config       = arguments.config
darkf        = arguments.darkf
flatf        = arguments.flatf
x0           = arguments.x0
y0           = arguments.y0
selfcalib    = arguments.selfcalib
ceniters     = arguments.ceniters
boxsize      = arguments.boxsize
windowradius = arguments.windowradius
windowfwhm   = arguments.windowfwhm
refimagef    = arguments.refimagef
edgewidth    = arguments.edgewidth
boxargs      = arguments.boxargs
refboxargs      = arguments.refboxargs
box1x        = arguments.box1x
box1y        = arguments.box1y
box2x        = arguments.box2x
box2y        = arguments.box2y
box3x        = arguments.box3x
box3y        = arguments.box3y
summarize    = arguments.summarize


all_dfspistons = []
all_dhspistons = []
all_ktpistons  = []

# Read and process dark
#
if (darkf!=None):
    dark = fits.open(darkf)[0].data
    if (len(dark.shape) == 3):
        darkav = dark.mean(axis=0)
    else:
        darkav = dark
        
    if selfcalib:
        dark -= dark.mean(axis=0)

# Read and process flat
#
if (flatf!=None):
    flat = fits.open(flatf)[0].data
    if (darkf != None):
        flat -= darkav

    bad_pix = np.where(flat <= 0)
    if (len(bad_pix) > 0):
        eprint("Fixing bad pixels in flat frame")
        flat[bad_pix] = 1.0

# Self calibrating mode is the mode Brian described via Skype where we
# subtract the mean of each pixel prior to processing in the normal manner.

# If self calibrating, apply flat to dark so the normalization is the same
#
if (selfcalib and darkf!=None and flatf!=None):
    dark /= flat


# Read reference images first so that we can use it as a baseline    
hdulist_ref = fits.open(refimagef)
primary_hdu_ref = hdulist_ref[0]
refimage = primary_hdu_ref.data
refhdr = primary_hdu_ref.header


# If using simulated fringes as baseline, populate arrays with known positions
# of fringes in frame.  Else, determine approximage fringe positions from
# FITS keywords of datafile, and then tweak via iterative centroiding.
if (str.split(refimagef,'/')[-1] == 'simulatedfringes.fits'):
    eprint("Using simulated reference")
    r0 = 62
    rref = [0,r0,r0]
    aref = np.array([0, 60, 120])
    xcenref = (188 + rref * np.cos(np.radians(aref))).astype(int)
    ycenref = (101 + rref * np.sin(np.radians(aref))).astype(int)
    # The simulated image fringes correspond to the 0', 8', and 6' OA positions
    oadistsref = np.array([0,8,6],dtype=np.int32)
else:
    eprint("Using measured reference")
    xcenref = []
    ycenref = []
    oadistsref = []

    # Get fringe postions in image for each probed off axis angle measured
    for box_loop in range(3):
        (xpos,ypos,oadist) = get_ref_fringe_config(refhdr,box_loop+1)
        xcenref.append(int(np.round(xpos)))
        ycenref.append(int(np.round(ypos)))
        oadistsref.append(int(np.round(oadist)))
    xcenref = np.array(xcenref)
    ycenref = np.array(ycenref)
    oadistsref = np.array(oadistsref)

    if refboxargs:
        xcenref = np.array([box1x, box2x, box3x]).astype(int)
        ycenref = np.array([box1y, box2y, box3y]).astype(int)
        

    # Apply darks and flats to ref image if it is measured
    if (darkf!=None):
        refimage -= darkav
    if (flatf!=None):
        refimage /= flat

    if (len(refimage.shape) == 3):
        imav = refimage.mean(axis=0)
    else:
        imav = refimage

    # Iteratively find the center of the reference fringes
    b2 = int(boxsize/4)

    
    for k in range(3):
        subim = imav[ycenref[k]-b2:ycenref[k]+b2,
                     xcenref[k]-b2:xcenref[k]+b2].copy()

        
        for nit in range(ceniters):
            cent = center_of_mass(subim)
            ycenref[k] += np.round(cent[0]-b2+0.5)
            xcenref[k] += np.round(cent[1]-b2+0.5)
#            print("%i %i %f %f %f %f" % (k,nit,cent[0],cent[1],ycenref[k],xcenref[k]))

            subim = imav[ycenref[k]-b2:ycenref[k]+b2,
                         xcenref[k]-b2:xcenref[k]+b2].copy()
#            hdu = fits.PrimaryHDU(subim)
#            hdul = fits.HDUList([hdu])
#            hdul.writeto("ceniterimg_%i_%i.fits" % (k,nit),overwrite=True)

            bkgnd = np.mean((subim[0,:],subim[-1,:],subim[:,0],subim[:,-1]))
            subim -= bkgnd

# Extract subimages with fringes, and compute FFTs for each of the three
# fringes
ref = PhaseData(refimage, xcenref, ycenref, boxsize, edgewidth)



# Compute analytic FFT of pinhole source.  It is not a true point
# source, so we want to convolve the data with this slighty extended source
pfft = pinholefft(pinhole_rad_pix, boxsize)

# If using simulated reference fringes that are compute with a perfect point
# source, convolve simulated fringe image with pinhole by multiplying FFTs
if (str.split(refimagef,'/')[-1] == 'simulatedfringes.fits'):
    # Convolve all fringes with extended source
    print("Convolving fringes with extended pinhole source")
    for blur_loop in range (len(refffts)):
        ref.ftabs[blur_loop] = refffts[blur_loop] * pfft

ref.savefits("ref")

# Compare reference FFTs with themselves to get nominal contrast and peak
# positions.  All normalization comes "out in the wash" and you don't have
# to think too much about it according to Brian.
ref.computeresults(ref, ccmin, ccmax, PixPerNm, Wavelength)


# Look through all data files passed in one by one
for dataf in args:
    # Open file
    hdulist = fits.open(dataf)
    primary_hdu = hdulist[0]
    data = primary_hdu.data
    hdr = primary_hdu.header

    # We want all arrays to be 3D even if we only get one frame
    if (len(data.shape) == 3):
        imav = data.mean(axis=0)
    else:
        imav = data

    if selfcalib:
        datamean = data.mean(axis=0)
        data = data - datamean
    elif (darkf!=None):
        data -= dark


    if (flatf!=None):
        data /= flat

    if (darkf!=None):
        imav = imav - darkav

    # Probe FITS headers to determine where the fringes are located in the frame
    # and which off-axis angles are currently configured (mask and prism pos
    # dependent)

    if 'MASKNAME' in hdr.keys():     
        xcenimg = []
        ycenimg = []
        oadistsimg = []
        for box_loop in range(3):
            (xpos,ypos,oadist) = get_fringe_config(hdr,box_loop+1)
            xcenimg.append(int(np.round(xpos)))
            ycenimg.append(int(np.round(ypos)))
            oadistsimg.append(int(np.round(oadist)))

    # Get the box positions off the command line if told to
    if boxargs:
        xcenimg = [box1x, box2x, box3x]
        ycenimg = [box1y, box2y, box3y]

    # Config file has these as lists, so cast to arrays
    xcenimg = np.array(xcenimg,dtype=np.int)
    ycenimg = np.array(ycenimg,dtype=np.int)
    oadistsimg = np.array(oadistsimg)

    # Perform median of 3x3 regions to remove lingering hot pixels from
    # centroid based box position optimization.  Needed in low SNR cases
    imav = scipy.signal.medfilt(imav,3)
    imav -= np.median(imav)


    shiftx, shifty = proto3_centers(imav, hdr)

    
    # Iteratively find the center of the fringes
    b2 = int(boxsize/4)
    fwhmest = np.zeros(3)
    for k in range(3):
        xcenimg[k] += shiftx
        ycenimg[k] += shifty
        eprint("Box %d: %d %d" % (k, xcenimg[k], ycenimg[k]))
        subim = imav[ycenimg[k]-b2:ycenimg[k]+b2,
                     xcenimg[k]-b2:xcenimg[k]+b2].copy()

        for nit in range(ceniters):
            cent = center_of_mass(subim)
            ycenimg[k] += np.round(cent[0]-b2+0.5)
            xcenimg[k] += np.round(cent[1]-b2+0.5)
#            print("%i %i %f %f %f %f" % (k,nit,cent[0],cent[1],ycenref[k],xcenref[k]))

            subim = imav[ycenimg[k]-b2:ycenimg[k]+b2,
                         xcenimg[k]-b2:xcenimg[k]+b2].copy()

            bkgnd = np.mean((subim[0,:],subim[-1,:],subim[:,0],subim[:,-1]))
            subim -= bkgnd
        fits.PrimaryHDU(subim).writeto('box%d.fits'%(k+1),overwrite=True)

        # fwhm is calculated by estimating the number of pixels with a greater
        # intensity than half the maximum.

        # This got broken when copying from Marcos script.
        # Maybe he can get this fixed
        # npixgthalf = len(np.where(subim >= np.max(subim)/2)[0])
        # 
        # fwhmest[k] = np.sqrt(4.*npixgthalf/np.pi)*pixsize;

    # Compute FFTs of dark images for better removal of dark noise in selfcalib
    # mode.
    if selfcalib:
        dark = PhaseData(dark,xcenimg,ycenimg,boxsize, edgewidth)

    # Compute FFTs of fringes
    eprint ("data", data.shape)
    eprint ("xcenimg, ycenimg", xcenimg, ycenimg)
    eprint ("boxsize", boxsize)
    eprint ("refboxes", ref.boxavg[0].shape)
    
    im = PhaseData(data, xcenimg, ycenimg, boxsize, edgewidth, refboxes = ref.boxavg)

    # Loop over three fringe pattern FFTs and perfrom dark subtraction in
    # selfcalib mode, and save FFTs as FITS

    import pyds9
    ds9 = pyds9.DS9()
    ds9.set('frame 1')
    ds9.set_np2arr(im.dfsabs[0])
    if selfcalib:
        ds9.set('frame 2')
        ds9.set_np2arr(dark.dfsabs[0])

    for i in range(len(im.dfsabs)):
        if selfcalib:
            # Seems like it would be more correct to subtract the dark noise
            # in quadrature but it doesnt work very well
            #  temp = np.sqrt(np.abs(im.dfsabs[i]**2 - darkfftlist[i]**2 ))
            temp = im.dfsabs[i]**2 - dark.dfsabs[i]**2 
            #temp = im.dfsabs[i] - dark.dfsabs[i]
            im.dfsabs[i] = temp

        ds9.set('frame 3')
        ds9.set_np2arr(im.dfsabs[0])
            
        fftname = '.fft'.join(os.path.splitext(dataf))
        fits.PrimaryHDU(np.array(im.dfsabs)).writeto(fftname,overwrite=True)
        dphixname = '.dphix'.join(os.path.splitext(dataf))
        fits.PrimaryHDU(np.array(im.dxphi)).writeto(dphixname,overwrite=True)
        dphiyname = '.dphiy'.join(os.path.splitext(dataf))
        fits.PrimaryHDU(np.array(im.dyphi)).writeto(dphiyname,overwrite=True)
        boxname = '.box'.join(os.path.splitext(dataf))
        fits.PrimaryHDU(np.array(im.boxavg)).writeto(boxname,overwrite=True)

    if (str.split(refimagef,'/')[-1] == 'simulatedfringes.fits'):
        # Shuffle analyzed reference FFTs to match order of boxes in measured
        # data
        reordered_contrastrefs = []
        reordered_peakrefs = []
        saveft = ref.dfsabs.copy()
        savecontrasts = ref.contrasts.copy()
        saveypeaks = ref.ypeaks.copy()
        savexpeaks = ref.xpeaks.copy()
        ref.dfsabs = []
        ref.constrasts = []
        ref.ypeaks = []
        ref.xpeaks = []
        for oaindex in range(len(oadistsimg)):
            data_oapos = oadistsimg[oaindex]
            if (data_oapos == 0):
                ref.dfsabs.append(saveft[0])
                ref.constrasts.append(savecontrasts[0])
                ref.ypeaks.append(saveypeaks[0])
                ref.xpeaks.append(savexpeaks[0])
            elif (data_oapos == 6):
                ref.dfsabs.append(saveft[2])
                ref.constrasts.append(savecontrasts[2])
                ref.ypeaks.append(saveypeaks[2])
                ref.xpeaks.append(savexpeaks[2])
            elif (data_oapos == 8):
                ref.dfsabs.append(saveft[1])
                ref.constrasts.append(savecontrasts[1])
                ref.ypeaks.append(saveypeaks[1])
                ref.xpeaks.append(savexpeaks[1])

    # Find peak locations in Fourier domain, and get measured contrasts, DFS, DHS, Knox-Thompson piston estimates
    im.computeresults(ref, ccmin, ccmax, PixPerNm, Wavelength)
    
    # Normalize contrast by reference contrast
    contrasts  = im.contrasts / ref.contrasts

    dfspistons = (im.dfspiston - ref.dfspiston )/1000.  
    dhspistons = im.dhspiston - ref.dhspiston
    ktpistons  = (im.ktpiston  - ref.ktpiston) * -1

    all_dfspistons.append(dfspistons)
    all_dhspistons.append(dhspistons)
    all_ktpistons.append(ktpistons)
    
    print("{0:32s}  Contrasts: {4:.2f} {5:.2f} {6:.2f} Centroids: {7:d} {8:d} {9:d} {10:d} {11:d} {12:d} DFS: {1:.3f} {2:.3f} {3:.3f} DHS: {13:.3f} {14:.3f} {15:.3f} KT: {16:.3f} {17:.3f} {18:.3f}".format(
        str.split(dataf,'/')[-1],
        dfspistons[0], dfspistons[1], dfspistons[2],
        contrasts[0], contrasts[1], contrasts[2],
        xcenimg[0], ycenimg[0], xcenimg[1], ycenimg[1], xcenimg[2], ycenimg[2],
        dhspistons[0], dhspistons[1], dhspistons[2],
        ktpistons[0], ktpistons[1], ktpistons[2]))

    
    im.savefits('obj')

if summarize:
    std_dfspistons = np.std(all_dfspistons, axis=0)
    std_dhspistons = np.std(all_dhspistons, axis=0)
    std_ktpistons =  np.std(all_ktpistons, axis=0)
    mean_dfspistons = np.mean(all_dfspistons, axis=0)
    mean_dhspistons = np.mean(all_dhspistons, axis=0)
    mean_ktpistons =  np.mean(all_ktpistons, axis=0)

    print ("Summary")
    print ("DFS: ",end="")
    [ print ("  {0:6.3f} +/- {1:5.3f}".format(mean_dfspistons[i], std_dfspistons[i]), end="")  for i in [0,1,2] ]
    print ("")        

    print ("DHS: ",end="")
    [ print ("  {0:6.3f} +/- {1:5.3f}".format(mean_dhspistons[i], std_dhspistons[i]), end="")  for i in [0,1,2] ]
    print ("")        
        
    print ("KT:  ",end="")
    [ print ("  {0:6.3f} +/- {1:5.3f}".format(mean_ktpistons[i], std_ktpistons[i]), end="")  for i in [0,1,2] ]
    print ("")        
        
                    
