import numpy as np

class DFS:

    def __init__(self, pixsize=0.07, dispersion=5, nsteps=40, wavemin=1.027e-6, wavemax=1.358e-6, pupilpix=0.25,fieldang=0, imsize = 48, gap=0.4):
        """Initialize a DFS

        Arguments:

        dispersion:  in arcsec per micron
        pixsize: in arcsec
        pupilpix: in meters
        

        """
        Dpri = 8.4

        npix = int(wavemax * 206265 / pixsize / pupilpix)

        Apsize = 1.5

        pupil=np.zeros((npix,npix))
        ypix,xpix=np.indices(pupil.shape,dtype=np.float32) - npix/2
        y = ypix * pupilpix
        x = xpix * pupilpix
        rpix = np.sqrt(xpix*xpix+ypix*ypix)

        x0 =  (Dpri+gap) / 2
        x1 = -(Dpri+gap) / 2
        rad2 = Dpri*Dpri / 4

        mask1 = ((x-x0)*(x-x0) + y*y < rad2)
        mask2 = ((x-x1)*(x-x1) + y*y < rad2)
        mask3 = (abs(x) < Apsize/2) * (abs(y) < Apsize/2)
        mask = (mask1 + mask2) * mask3

        pupil[mask] = 1.
        self.waves = np.linspace(wavemin,wavemax,nsteps)
        self.wave0 = (wavemin + wavemax) / 2.
        self.pupil = pupil
        self.ymeters = y
        self.xmeters = x
        self.dispersion = dispersion
        self.imagepixsize = pixsize
        self.pupilpixsize = pupilpix
        self.npix = npix
        self.imsize = imsize

        
    def mkimage(self,wavelen=1.2e-6,piston=0e-6,wavefrontmap=None):

        if wavefrontmap is not None:
            wfnpix = wavefrontmap.shape[0]
            wfpix0 = self.npix // 2 - wfnpix //2

        self.image = np.zeros((self.imsize,self.imsize))

        # Loop over wavelengths and sum up the output image
        #
        for wave in self.waves:

            # The size of the FFT determines the output pixel size
            npix = int(wave * 206265 / self.imagepixsize / self.pupilpixsize)
            pix0 = (self.npix - npix) // 2

            # Now build up the wavefront 
            # First the tilt due to the prism dispersion
            wavefront = -1 * (wave - self.wave0) * self.dispersion * 1.0e6 / 206265. * self.ymeters

            # Add in externally supplied wavefront
            if wavefrontmap is not None:
                wavefront[wfpix0:wfpix0+wfnpix,wfpix0:wfpix0+wfnpix] += wavefrontmap

            # Add piston to one half of aperture
            wavefront[self.xmeters>0] += piston

            # Compute the FFT of the pupil
            cmplx = np.fft.fftshift(np.fft.fft2((self.pupil * np.exp(1j * wavefront / wave * 2 * np.pi))[pix0:pix0+npix,pix0:pix0+npix]))
            impix0 = npix//2 - self.imsize//2

            # Image is square of amplitude of FFT
            # Extract imsize x imsize box from the center
            self.image += (cmplx.real**2 + cmplx.imag**2)[impix0:impix0+self.imsize,impix0:impix0+self.imsize]
            
        return self.image
        

if (__name__ == "__main__"):

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

    # Conversion of pixel shift of FFT peak to nm of piston error
    PixPerNm = boxsize  / PixelsPerMicron / Wavelength**2 / 1000.

#   Generate reference image
    d = DFS(pupilpix=0.025,imsize=imsize, pixsize = pixsize, dispersion = dispersion)
    refimage = d.mkimage()

#   Measure reference image
    ref = PhaseData(refimage, [imsize/2], [imsize/2], boxsize, edgewidth)
    ref.computeresults(ref, ccmin, ccmax, PixPerNm, Wavelength)
    
#   Check gain

    inputs = []
    outputs = []

    for piston in np.linspace(-10e-6,10e-6,51):
        
        image = d.mkimage(piston=piston)
        im = PhaseData(image, [imsize/2], [imsize/2], boxsize, edgewidth, refboxes=ref.boxavg)
        im.computeresults(ref, ccmin, ccmax, PixPerNm, Wavelength)

        measuredpiston = -(im.dfspiston - ref.dfspiston)
        print (piston * 1e9, measuredpiston)

        inputs.append(piston*1e9)
        outputs.append(measuredpiston[0])

    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)
    
    import pylab as pl
    pl.plot(inputs, outputs-inputs,'.-')
    pl.xlabel("Piston input to simulation")
    pl.ylabel("Measured - Input")

    pl.figure()
    pl.plot(inputs, outputs,'.-')
    pl.plot(inputs, inputs, '-')
    pl.xlabel("Piston input to simulation")
    pl.ylabel("Measured piston")

    
    pl.show()
        
        

    


