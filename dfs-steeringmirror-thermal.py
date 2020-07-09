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
d = DFS(pupilpix=0.025,imsize=imsize, pixsize = pixsize, dispersion = dispersion)
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

# OK, now get to work and read the 

