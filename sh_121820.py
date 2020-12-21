from astropy.io import fits
import numpy as np
import pylab as pl
%matplotlib notebook

import glob
import sh
import zernike
import re

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom, rotate
import glob
import matplotlib.pyplot as plt
from shanalyzefull import shgridcenter
from sh_analysisfxns_pb import raw_to_npy, get_spots

# shwfdir = '/home/pbenac/home/Thesis/raw_SHWF_images'
filelist = glob.glob(shwfdir + '/SHWF-11132020*.raw')
filelist.sort()
[data, images] = raw_to_npy(filelist)


# import pyds9
# ds9 = pyds9.DS9('ds9')
# ds9.set_np2arr(images[1])


# Pixel scale
# 1.5m @ f/10.4 --> ???
# 53mm lenslet --> 26.831
# 56mm collimator --> 97.158 
# 20 um pixels --> 16um
pixscale = 206265/(1500*10.4) / 26.831 * 97.158 * 0.016

centroiding_threshold = 0

flux_frac = 0.4
shlist = []
# frames = ds9.get("frame all").split()
# print ("                      HA Dc Ast45  Ast0   ComaX   ComaY   Sphr TrefX   TrefY  High  Tot ")
allzerns = []
allha = []
alldec = []

for image in images:
#     ds9.set('frame ' + frame)
#     file = ds9.get('file')
    
    shwfs = sh.SH(image, pixscale, flux_frac)
    shlist.append(shwfs)
    shwfs.process(nzern=10)
    
    # Parse the HA and Dec out of the file name
    ha_dec_str = file.split('.')[2]
    ha_dec_str = re.sub('[p]',' ',ha_dec_str)
    ha_dec_str = re.sub('[m]',' -',ha_dec_str)
    x,ha,dec = ha_dec_str.split(' ')


    print (file, ha, dec, shwfs.zcoeffs[4:] * shwfs.pixscale, "%.2f %.2f" %(shwfs.rms, shwfs.rms_ttf))

    # Draw the spot locations and error vectors on ds9
#     s = ""
#     ds9.set('regions delete all')
#     for x,y,xv,yv,xvho,yvho in zip(shwfs.xrefs_ttf+1,shwfs.yrefs_ttf+1,shwfs.xvec,shwfs.yvec,shwfs.xvec_ho,shwfs.yvec_ho):
#         s = s + "circle(%f,%f,1) # color=cyan\n" % (x,y)
#         s = s + "circle(%f,%f,5)\n" % (x+xv,y+yv)
#         s = s + "line(%f,%f,%f,%f) #line=0 1 color=yellow\n "% (x,y,x+xv*10,y+yv*10)
#         s = s + "line(%f,%f,%f,%f) #line=0 1 color=red\n "% (x,y,x+xvho*10,y+yvho*10)
#     ds9.set('regions',s)

    allzerns.append(shwfs.zcoeffs * shwfs.pixscale)
    allha.append(float(ha))
    alldec.append(float(dec))
    
allzerns = np.asarray(allzerns)