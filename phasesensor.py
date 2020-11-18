#!/usr/bin/python3
import astropy.io.fits as fits
import numpy as np
import pylab as pl
from scipy.signal import correlate
from scipy.ndimage.measurements import center_of_mass
import scipy.special
from scipy import ndimage

recenter_with_xcor = False

class PhaseData:

    def __init__(self,indata, xcen, ycen, n, edgewidth,windowradius=14, windowfwhm=4, startindex=None, stopindex=None, refboxes=None):
        """Compute and coadd the FFTs of a list of boxes extracted from a cube of images

        Arguments:
           indata:       2d image [y,x] or 3d image cube [frame, y, x] or 4d cube [frame, box, y, x]
           xcen, ycen:   lists giving the locations in the input images about which to extract smaller boxes
           n:            size of box to cut out
           edgewidth:    size border at edge of box used for computing background level
           windowradius: size of circular window used to truncate data
           windowfwhm:   gaussian fwhm used to blur window tophat function
           startindex:   python index of first frame to process
           stopindex:    python index of last frame to process

        Attributes: (each of these is a list, one item per box position)
           dfsabs: sum of absolute value of fft of each image in the cube
           ftphi: phase of sum of fft of each image
           ftabs: abs of sum of fft of each image
           dxphi: phase of ft(v,u) x ft*(v,u+1) -- Knox-Thompson cross term x component
           dyphi: phase of ft(v,u) x ft*(v+1,u) -- Knox-Thompson cross term y component
           dxabs: amplitude of Knox-Thompson cross term x component
           dyabs: amplitude of Knox-Thompson cross term y component
           boxavg: mean of each box 
        """

        ## apply a soft circular window

        b2 = n/2
        boxsize = n

        xcoordinates = np.arange(0, boxsize, 1)-b2+0.5
        ycoordinates = xcoordinates[:,np.newaxis]
        radialdist2 =  (xcoordinates**2 + ycoordinates**2)

        hardwindow = np.zeros((boxsize,boxsize))
        hardwindow[np.where(radialdist2 <= windowradius**2)] = 1
        gaussiankernel = np.exp(-4*np.log(2)*radialdist2/windowfwhm**2)

        softwindow = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(hardwindow)*np.fft.fft2(np.flipud(np.fliplr(gaussiankernel))))))
        softwindow /= np.max(softwindow)

        self.dfsabs = []
        self.ftabs = []
        self.ftphi = []
        self.dxphi = []
        self.dyphi = []
        self.dxabs = []
        self.dyabs = []
        self.boxavg = []
        self.dfsabs = []

        boxy,boxx=np.indices((n,n))

        edge = (boxy<edgewidth) + (boxy>=n-edgewidth) + (boxx<edgewidth)  + (boxx>=n-edgewidth)

        i=1
        iy,ix = np.indices((n,n)) - n / 2.
        # Iterate over each box in the input center arrays
        for ibox,(x,y) in enumerate(zip(xcen,ycen)):

            if (y<n/2):
                print ('Too close to the edge  %d %d' % (x,y))

            if (len(indata.shape) == 2):
                data = np.expand_dims(indata[startindex:stopindex],0);
            elif(len(indata.shape) == 4):
                data = indata[startindex:stopindex,ibox,:,:]
            elif(len(indata.shape) == 3):
                data = indata[startindex:stopindex]

            dfsabs = np.zeros((n,n))
            ft = np.zeros((n,n), dtype=np.complex_)
            boxavg = np.zeros((n,n))
            dxsum = np.zeros((n,n-1), dtype=np.complex_)
            dysum = np.zeros((n-1,n), dtype=np.complex_)

            # Prepare for recentering using cross correlation 
            if refboxes!=None:
                reffft_conj = np.conj(np.fft.fftshift(np.fft.fft2(refboxes[ibox])))

            peakxx = []
            peakyy = []

            # For each short exposure image
            #
            for im in data:
                    
                # Extract the box
                box = im[int(y-n/2):int(y+n/2),int(x-n/2):int(x+n/2)].copy()

                # Subtract the background level using the edge pixels
                box = box - np.mean(box[edge])

                # Compute the FFT
                boxfft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(box*softwindow)))
                
                if refboxes != None and recenter_with_xcor:
                    # Recenter by cross-correlating with template
                    xcor = np.real(np.fft.fft2(np.fft.fftshift(boxfft * reffft_conj)))
                    # Peak pixel in cross-correlation
                    peakpix = np.unravel_index(np.argmax(xcor), np.shape(xcor))

                    # Interpolated peak
                    # Y
                    I1 = xcor[peakpix[0] - 1, peakpix[1]]
                    I2 = xcor[peakpix[0],     peakpix[1]]
                    I3 = xcor[peakpix[0] + 1, peakpix[1]]
                    peaky = peakpix[0] + (I1 - I3) / ( I1 + I3 - 2 * I2 ) / 2 - n/2
                    
                    # X
                    I1 = xcor[peakpix[0], peakpix[1] - 1]
                    I3 = xcor[peakpix[0], peakpix[1] + 1]
                    peakx = peakpix[1] + (I1 - I3) / ( I1 + I3 - 2 * I2 ) / 2 - n/2

                    # Shift image by adjusting phase 
                    boxfft = boxfft * np.exp(-(ix * peakx  + iy * peaky) / n * 2 * np.pi * 1j)
                
                # Sum KT cross terms and abs(FFT)
                dxsum = dxsum + boxfft[:,:-1] * np.conj(boxfft)[:,1:]
                dysum = dysum + boxfft[:-1,:] * np.conj(boxfft)[1:,:]
                dfsabs = dfsabs + abs(boxfft)
                self.box = box
                self.boxfft = abs(boxfft)
                # Sum FT for DHS analysis
                ft = ft + boxfft

                boxavg += box * softwindow

            # Use shifted sum 
            if refboxes!=None:
                boxavg = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ft))))

            # Normalized sum(abs(FFT))
            n2=int(n/2)
            dfsabs = dfsabs / (dfsabs[n2-1,n2] + dfsabs[n2+1,n2] + dfsabs[n2,n2-1] + dfsabs[n2,n2+1])
            
            # Normalize mean image
            boxavg = boxavg / len(data)
            #
            # Compute centroid of mean image
            xcen = (ix * boxavg).sum() / boxavg.sum()
            ycen = (iy * boxavg).sum() / boxavg.sum()

            # Remove wavefront tilt using measured centroid
            centroid_phase_correction = -(ix * xcen  + iy * ycen) / n * 2 * np.pi 
            centroid_dx_correction = -xcen / n * 2 * np.pi
            centroid_dy_correction = -ycen / n * 2 * np.pi
            
            self.dfsabs.append(dfsabs)
            self.ftphi.append(np.angle(ft) - centroid_phase_correction)
            self.ftabs.append(np.abs(ft))
            self.dxphi.append(np.angle(dxsum) - centroid_dx_correction)
            self.dyphi.append(np.angle(dysum) - centroid_dy_correction)
            self.dxabs.append(np.abs(dxsum))
            self.dyabs.append(np.abs(dysum))
            self.boxavg.append(boxavg)

    def computeresults(self, ref, ccmin, ccmax, PixPerNm, Wavelen):
            contrastrefs, xpeakrefs, ypeakrefs = ref.analyzefft( ref, ccmin, ccmax)
            contrastobjs, xpeakobjs, ypeakobjs = self.analyzefft( ref, ccmin, ccmax)
#            print ("Peakref: ", ypeakrefs, "Peakobj: ", ypeakobjs)
            self.dfspiston = (np.array(ypeakobjs) - np.array(ypeakrefs)).flatten() / PixPerNm
            self.dhspiston = []
            self.ktpiston  = []

            for i in range(len(self.ftphi)):
#                self.dhspiston.append((self.ftphi[i][int(ypeakobjs[i]),int(xpeakobjs[i])]-ref.ftphi[i][int(ypeakrefs[i]),int(xpeakrefs[i])]) / 2 / np.pi * Wavelen)
                self.dhspiston.append(self.dhsanalyze(self.ftphi[i], 1, 5,12, Wavelen))
                dxphi_diff = self.dxphi[i][20,:] - ref.dxphi[i][20,:]
                dxphi_diff -= dxphi_diff[20:24].mean()
                self.ktpiston.append(dxphi_diff[24:27].sum() / np.pi / 2. * Wavelen)

            self.dfspiston = np.array(self.dfspiston)
            self.dhspiston = np.array(self.dhspiston)
            self.ktpiston = np.array(self.ktpiston)

    def dhsanalyze(self,phasemap,maxrow,col_mid,col_end,Wavelen):
        n = len(phasemap)
        # Assume that phases are within +/- pi
        vec = phasemap[n//2 - maxrow : n//2 + maxrow + 1 , n//2 - col_end : n//2 + col_end + 1 ].mean(axis=0)
        A=np.zeros((2,len(vec)))
        nvec = len(vec)
        A[0] = np.arange(len(vec)) - nvec//2
        A[1][:nvec//2-col_mid] = -1
        A[1][nvec//2+col_mid+1:] = 1
        fit = np.linalg.lstsq(A.T,vec, rcond=None)[0]  
        vec_fit = fit.dot(A)
        phase_err  = fit[1]/ np.pi / 2. * Wavelen
        return phase_err
    
    def analyzefft(image, kernel, colmin, colmax):

        frngpows = []
        xpeaks = []
        ypeaks = []

        if not isinstance(kernel, PhaseData):
            raise TypeError('Kernel must be a PhaseData object')
        if not isinstance(image, PhaseData):
            raise TypeError('Image must be a PhaseData object')
            
        imageffts = image.dfsabs
        kernellist = kernel.dfsabs

        if (len(imageffts) != len(kernellist)):
            raise Error("You need to pass in a kernel for each box")
            

        for box_loop in range(len(imageffts)):
            z = imageffts[box_loop]
            kernel = kernellist[box_loop].copy()

            yi,xi=np.indices(kernel.shape)
            n = kernel.shape[0]
            kernel[(xi<colmin) | (xi>colmax)] = 0

            cc = np.real(correlate(z,kernel[int(n/2-n/8):int(n/2+n/8),:], mode='valid'))

            maxindex = np.argmax(cc)
            cx = np.arange(cc.size)

            # Fit a parabola to the central 3 pixels, and determine the maximum
            #
            center=(abs(cx-maxindex)<2)
            p = np.polyfit(cx[center],cc[center],2)

            fit = p[0] * cx * cx + p[1] * cx + p[2]

            ypeak =  -p[1]/2./p[0]

            peakval = p[0] * ypeak * ypeak + p[1] * ypeak + p[2]

            n2 = int(n/2)

            frngpow = float(peakval / (z[n2-1,n2] + z[n2+1,n2] + z[n2,n2-1] + z[n2,n2+1]))

            xpeaks.append(np.argmax(kernel.sum(axis=0)))
    #        print('ypeak %f %f' % ( ypeak, ypeak+n/4))
            ypeaks.append(ypeak + n/8)
            frngpows.append(frngpow)
        image.contrasts = np.array(frngpows)
        image.xpeaks = np.array(xpeaks)
        image.ypeaks = np.array(ypeaks)
        return frngpows, xpeaks, ypeaks

    def savefits(self, root):
        fits.PrimaryHDU(self.ftabs).writeto(root+"-ftabs.fits",overwrite=True)
        fits.PrimaryHDU(self.ftphi).writeto(root+"-ftphi.fits",overwrite=True)
        fits.PrimaryHDU(self.dfsabs).writeto(root+"-dfsabs.fits",overwrite=True)
        fits.PrimaryHDU(self.boxavg).writeto(root+"-boxes.fits",overwrite=True)
        fits.PrimaryHDU(self.dxphi).writeto(root+"-dphix.fits",overwrite=True)
        fits.PrimaryHDU(self.dxphi).writeto(root+"-dphiy.fits",overwrite=True)


def pinholefft(rad, n):
    # Compute FFT of pinhole circular tophat = sombrero function

    y,x=np.indices((n,n))
    y = y - n/2
    x = x - n/2
    r = np.sqrt((y**2+x**2))
    rho = 2 * np.pi * r * rad / n
    rho[int(n/2),int(n/2)] = 1 # Avoid dividing by zero
    b = 2 * abs(scipy.special.jn(1,rho)/rho) 
    b[int(n/2),int(n/2)] = 1
    return b


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
    upper_box1_xcenter = 203
    upper_box1_ycenter = 193

    upper_box2_xcenter = 234
    upper_box2_ycenter = 143

    upper_box3_xcenter = 177
    upper_box3_ycenter = 143
    
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
    upper_box1_xcenter = 203
    upper_box1_ycenter = 193

    upper_box2_xcenter = 234
    upper_box2_ycenter = 143

    upper_box3_xcenter = 177
    upper_box3_ycenter = 143

    
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
                
