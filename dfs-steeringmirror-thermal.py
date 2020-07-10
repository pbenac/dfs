from dfssim import *
from  phasesensor import *
import pyds9
import time
ds9 = pyds9.DS9()
ds9.set('frame 1')
wavefrontmap = np.zeros((80,80))
wavefrontmap[:,40:] = 1

boxsize = 40
imsize = 100
edgewidth = 8
ccmin = int(53 * boxsize / 80)
ccmax = int(77 * boxsize / 80)

pupilpixsize = 0.025

# Spectral dispersion
pixsize    = 0.07 # arcsec
dispersion = 5 # arcsec per micron
PixelsPerMicron = dispersion / pixsize
Wavelength = ( 1.358 + 1.027 ) / 2.
UnityGain  = 1

# Conversion of pixel shift of FFT peak to nm of piston error
PixPerNm = boxsize  / PixelsPerMicron / Wavelength**2 / 1000.
print("PixPerNm (theory) = ", PixPerNm)

#   Generate reference image
d = DFS(pupilpix=pupilpixsize,imsize=imsize, pixsize = pixsize, dispersion = dispersion)
refimage = d.mkimage()

#   Measure gain with 200nm piston

#   Measure reference image
ref = PhaseData(refimage, [imsize/2], [imsize/2], boxsize, edgewidth)
ref.computeresults(ref, ccmin, ccmax, UnityGain, Wavelength)

piston = 200e-9

image = d.mkimage(piston=piston)
im = PhaseData(image, [imsize/2], [imsize/2], boxsize, edgewidth, refboxes=ref.boxavg)
im.computeresults(ref, ccmin, ccmax, UnityGain,  Wavelength)

measuredpiston = im.dfspiston - ref.dfspiston

print (piston * 1e9, measuredpiston)

PixPerNm = measuredpiston / ( piston * 1e9 )
print("PixPerNm (calibrated) = ", PixPerNm)

ref.computeresults(ref, ccmin, ccmax, PixPerNm,  Wavelength)
im.computeresults(ref, ccmin, ccmax, PixPerNm,  Wavelength)

measuredpiston = im.dfspiston - ref.dfspiston

print (piston * 1e9, measuredpiston)

# OK, now get to work and read the wavefront file
filename =  '/Users/bmcleod/GoogleDrive/GMT AGWS and Wavefront Control Programs/4 Phasing Prototype and AGWS Design Study (2015-present)/4.8 AGWS Design Study/4.8.7 STOP/FDR 2020/SM_Model163/Wavefront_map_all_sm_errors_model_163.txt'

# File header
with open(filename) as f:
    linelist = f.readlines()

wavelen = float(linelist[8].split()[0]) * 1e-6

steeringmirror = np.genfromtxt(filename, skip_header=16,unpack=True) * wavelen

dfs_apsize_pix = int(1.5 / pupilpixsize) 
pupil_fraction_to_use = 0.9

maxrad2 = (steeringmirror.shape[0] / 2 * pupil_fraction_to_use) ** 2
pistons = []
wfsrms = []
pvs = []

iy,ix = np.indices((dfs_apsize_pix+2, dfs_apsize_pix+2))
A = np.array([np.ones((dfs_apsize_pix+2,dfs_apsize_pix+2)), iy, ix]).reshape((3,-1))

residmap = np.zeros(steeringmirror.shape)

for j in np.arange(0,steeringmirror.shape[0],dfs_apsize_pix):
    for i in np.arange(0,steeringmirror.shape[1],dfs_apsize_pix):
        j0 = j - steeringmirror.shape[0] // 2
        i0 = i - steeringmirror.shape[1] // 2
        j1 = j0 + dfs_apsize_pix
        i1 = i0 + dfs_apsize_pix
        if (j0**2 + i0**2) > maxrad2 : continue
        if (j1**2 + i0**2) > maxrad2 : continue
        if (j0**2 + i1**2) > maxrad2 : continue
        if (j1**2 + i1**2) > maxrad2 : continue

        wf = steeringmirror[ j-1 : j+dfs_apsize_pix+1, i-1 : i+dfs_apsize_pix+1]
        # Remove tip/tilt
        fit = np.linalg.lstsq(A.T,wf.flatten(),rcond=None)[0]
        resid = wf.flatten() - fit.dot(A)

        residmap[ j-1 : j+dfs_apsize_pix+1, i-1 : i+dfs_apsize_pix+1] = resid.reshape(wf.shape)

        wfsrms.append( resid.std() )
        pv = np.amax(resid) - np.amin(resid)
        pvs.append(pv)
        
        image = d.mkimage(wavefrontmap=wf)
        im = PhaseData(image, [imsize/2], [imsize/2], boxsize, edgewidth, refboxes=ref.boxavg)
        im.computeresults(ref, ccmin, ccmax, PixPerNm,  Wavelength)

        pistons.append(im.dfspiston - ref.dfspiston)

ds9.set_np2arr(residmap)

pistons = np.asarray(pistons)
wfsrms = np.asarray(wfsrms)
pvs = np.asarray(pvs)

print ("Piston RMS: ", pistons.std())
print ("Mean of Subap RMS: ", wfsrms.mean() * 1e9)
print ("Mean PV: ", pvs.mean() * 1e9)



        



