import numpy as np
#import pylab as pl
import matplotlib.pyplot as pl
import zernike
import astropy.io.fits as fits

# Thresholded centroid
#
def centroid(imag, x, y, boxsize, thresh):
    ibx = int(boxsize)
    #print(boxsize)
    yi,xi=np.indices((ibx,ibx))
    box = imag[int(y-boxsize/2):int(y-boxsize/2)+ibx,int(x-boxsize/2):int(x-boxsize/2)+ibx].copy()
    #pl.imshow(box)
    #pl.show()
    edge = ((yi==0) + (yi==boxsize-1) + (xi==0) + (xi==boxsize-1))
#     print('edge', len(edge))
#     print('box', len(box))
    background = np.median(box[edge])
    box = box-background

    counts = box[box>thresh].sum()

    if counts>0:
        cenx = (box[box>thresh] * xi[box>thresh]).sum() / counts
        ceny = (box[box>thresh] * yi[box>thresh]).sum() / counts
        return int(x-boxsize/2)+cenx, int(y-boxsize/2)+ceny, counts
    else:
        return 0,0,0

# Determine mean spacing of spots in an axis
def shspacing(shimage,axis):

    ignore = 4

    oned = shimage.sum(axis=axis)

    xfft = np.fft.fft(oned) 
    xfft = np.real(xfft * np.conj(xfft))
    imax = np.argmax(xfft[ignore:len(xfft)//2]) + ignore 
    I1 = xfft[imax-1]
    I2 = xfft[imax]
    I3 = xfft[imax+1]
    peak =  imax + (I1 - I3) / (I1 + I3 - 2 * I2) / 2;
    spacing = len(xfft) / peak
    return spacing

# Find the edges of the pupil assuming we have a central obscuration
# by locating the places where the intensity crosses the 50% value
#
def computecenter(pslicey):
    #pl.figure()
    #pl.plot(pslicey)
    #center of pupil image
    nx = len(pslicey)
    psliceymax = np.amax(pslicey)
    dp = pslicey[1:] - pslicey[:-1]
    for thresh in [0.5,0.6,0.7,0.8,0.2]:
#         print('dp value', dp)
        dx = ( psliceymax * thresh - pslicey[1:] ) / dp
        crossings = abs(dx+0.5) < 0.5 
        crossers = np.where(crossings)[0]
        yvals = (dx+range(1,nx))[crossings]
        if len(crossers)==4:
            break

    if len(crossers)==4:
        pupildiay = yvals[3] - yvals[0]
        pupilobsc = yvals[2] - yvals[1]
        pupilceny1 = (yvals[3] + yvals[0]) / 2.
        pupilceny2 = (yvals[1] + yvals[2]) / 2.
    
        return(pupildiay, pupilobsc, (pupilceny1+pupilceny2)/2.)
    else:
        # Didn't find a central obscuration
        print('No central obscuration')
        ################
        if len(yvals) == 0 or len(yvals) == 1:
            return(0, 0, 0)
        # This pupildiay = 0 is what causes the zero division error.
        # pupildiay=1 causes ValueError
        #################
        else:
            pupildiay = yvals[1] - yvals[0]
            pupilobsc = 0
            pupilcen = (yvals[1] + yvals[0]) / 2.
            return(pupildiay, pupilobsc,pupilcen)

class SH:
    def __init__(self, img, pixscale=1, flux_frac = 0.5, centroiding_threshold=0, background=None, two_mirrors=False, pupil_radius_max = 1.0):
        if isinstance(img,np.ndarray):
            self.img = img
        elif isinstance(img,str):
            self.img = fits.open(img)[0].data.astype(float)

            
        if background is not None:
            if isinstance(background,np.ndarray):
                self.background = background
            elif isinstance(img,str):
                self.background = fits.open(background)[0].data
            self.img = self.img -  self.background

        self.pixscale = pixscale
        self.flux_frac = flux_frac
        self.centroiding_threshold = centroiding_threshold
        self.pupil_radius_max = pupil_radius_max
        
        self.dual = two_mirrors
    


        
    # Determine spacing and offset of an SH image
    # Save in self.spacings, self.offsets, each a 2-element array for x and y respectively
    #
    def shgridcenter(self, shimage):
        offset = np.zeros((2))
        spacing = np.zeros((2))
        for axis in [0,1]:
            spacing[axis] = shspacing(shimage,axis)
            gridspacing=np.int(np.round(spacing[axis]))

            cut = shimage.sum(axis=axis)
            center = len(cut) // 2
            centeroffset =  np.argmax(cut[center-gridspacing//2:center+gridspacing//2]) - gridspacing//2

            offset[axis] = center + centeroffset
            while offset[axis] >  2 * gridspacing:
                offset[axis] -= spacing[axis]
            offset[axis] -= spacing[axis]/2
        self.spacings = spacing
        self.offsets = offset

    # Compute the pupil coordinates and save in self.pupilx and self.pupily
    #
    def computepupilcoords(self):
        pupilmap = self.fluxmap
        ny,nx = pupilmap.shape

        nrows = 2
        x0 = nx / 2
        y0 = ny / 2
        for i in [0,1]:
            pslicex = pupilmap[int(y0-nrows):int(y0+nrows),:].sum(axis=0)
            pslicey = pupilmap[:,int(x0-nrows):int(x0+nrows)].sum(axis=1)
            pupildiax, pupilobscx, pupilcenx = computecenter(pslicex)
            pupildiay, pupilobscy, pupilceny = computecenter(pslicey)
            x0 = pupilcenx
            y0 = pupilceny

        py,px = np.indices(pupilmap.shape)
        pupildia = (pupildiax + pupildiay) / 2
        if pupildia != 0:
            self.pupilx = (px - pupilcenx) / (pupildia / 2.)
            self.pupily = (py - pupilceny) / (pupildia / 2.)
            self.obscureratio = (pupilobscx + pupilobscy) / 2 / pupildia
            self.pupilcenx = pupilcenx
            self.pupilceny = pupilceny
            self.pupildia = pupildia
            pr2 = self.pupilx**2 + self.pupily**2
            self.inpupil = (pr2 < 1) * (pr2 > self.obscureratio**2)
        else: # when pupildia=0
            self.pupilx = (px - pupilcenx) 
            self.pupily = (py - pupilceny)
            self.pupilcenx = pupilcenx
            self.pupilceny = pupilceny
            self.pupildia = pupildia
            
    
    def computegridofcentroids(self):
        xmin = self.offsets[0]
        ymin = self.offsets[1]
        img = self.img
        spacingx = self.spacings[0]
        spacingy = self.spacings[1]
        xmax = img.shape[1] - spacingx
        ymax = img.shape[0] - spacingy
        xcentroids = []
        ycentroids = []
        fluxes = []
        xrefs = []
        yrefs = []
        size_x = len(np.arange(xmin, xmax, spacingx))
        size_y = len(np.arange(ymin, ymax, spacingy))
        fluxmap = np.zeros((size_y,size_x))
        xcenmap = np.zeros((size_y,size_x))
        ycenmap = np.zeros((size_y,size_x))
        xrefmap = np.zeros((size_y,size_x))
        yrefmap = np.zeros((size_y,size_x))
        for index_x, x in enumerate(np.arange(xmin, xmax, spacingx)):
            for index_y,y in enumerate(np.arange(ymin, ymax, spacingy)):
                meanspacing = (spacingx + spacingy) / 2
                print('spacingx', spacingx)
                xcen, ycen, flux = centroid(img, x+spacingx/2, y+spacingy/2, round(spacingx), self.centroiding_threshold)
                fluxmap[index_y,index_x] = flux
                xcenmap[index_y,index_x] = xcen
                ycenmap[index_y,index_x] = ycen
                xrefmap[index_y,index_x] = xmin + index_x * meanspacing
                yrefmap[index_y,index_x] = ymin + index_y * meanspacing
        self.fluxmap = fluxmap
        self.xcenmap = xcenmap
        self.ycenmap = ycenmap
        self.xrefmap = xrefmap
        self.yrefmap = yrefmap
        # Make some plots

    def makeplots(self):
        img = self.img
        pl.figure()
        pl.imshow(self.fluxmap)
        pl.figure()
        pl.imshow(img)
        sx = self.spacings[0]
        ox = self.offsets[0]
        for x in np.arange(ox,len(img[0]),sx):
            pl.plot([x,x],[0,len(img)],'r',linewidth=1)
        sy = self.spacings[1]
        oy = self.offsets[1]
        for y in np.arange(oy,len(img),sy):
            pl.plot([0,len(img[0])],[y,y],'r',linewidth=1)
        pl.plot(self.xcentroids,self.ycentroids,'+')


    def process(self, nzern = 10, x1=None, y1=None, x2=None, y2=None):
        halist = []
        declist = []
        comaxlist = []
        comaylist = []
        trefxlist = []
        trefylist = []
        

        # Make a grid over the spots
        self.shgridcenter(self.img)
        
        # Compute centroids and fluxes of all points in the grid
        self.computegridofcentroids()
        
        fluxmap = self.fluxmap
        fluxthresh = np.amax(fluxmap) * self.flux_frac

        self.computepupilcoords()
        pupil_radius = np.sqrt(self.pupilx**2 + self.pupily**2)

        ## use something similar to select left,rght
#         if self.dual:
#             mir1_spots_to_use = (fluxmap>fluxthresh) #* (
#             mir2_spots_to_use = (fluxmap>fluxthresh)
#         else:#if not self.dual:
        spots_to_use = (fluxmap>fluxthresh) * (pupil_radius< self.pupil_radius_max)
        self.spots_to_use  = spots_to_use

        # Select the spots that exceed the flux threshold
        xcentroids = self.xcenmap[spots_to_use]
        ycentroids = self.ycenmap[spots_to_use]
        xrefs      = self.xrefmap[spots_to_use]
        yrefs      = self.yrefmap[spots_to_use]
        fluxes     = self.fluxmap[spots_to_use]

        def get_slope(x1, x2, y1, y2):
            m = (y2-y1) / (x2-x1)
            return m

        
        # Compute normalized pupil coordinates
        if self.dual:
            m = get_slope(x1, x2, y1, y2)
            y_intercept = (-m*x1) + y1
            mirror1_spots_to_use  = np.zeros(spots_to_use.shape)
            mirror2_spots_to_use  = np.zeros(spots_to_use.shape)

            for i, col in enumerate(self.xcenmap):
                for j, xcoord in enumerate(self.xcenmap[i]):
                    if self.spots_to_use[i][j]:
                        if self.ycenmap[i][j] <= (m*xcoord + y_intercept): # below or on the line
                            # MIRROR1
                            mirror1_spots_to_use[i][j] = 1
                        else: # Above the line
                            # MIRROR2
                            mirror2_spots_to_use[i][j] = 1
                            
            self.mirror1_spots_to_use = mirror1_spots_to_use
            self.mirror2_spots_to_use = mirror2_spots_to_use
            nsp = self.mirror1_spots_to_use.shape[0]
        pupilx = self.pupilx[spots_to_use]
        pupily = self.pupily[spots_to_use]
        #print("Pupilx ", pupilx)
        #print("Pupily ", pupily)
        # Subtract reference slopes
        slopex = xcentroids - xrefs
        slopey = ycentroids - yrefs

#         print('slopex shape', slopex.shape)
#         print('slopey shape', slopey.shape)
        # Fit Zernikes
        # Get the slopes of each zernike term at the pupil coordinates
        nz=nzern
        
        zerndx=[]
        zerndy=[]
        norms=[0]
        # Rotation of lenslet wrt pixels
        zerndx.append(pupily)
        zerndy.append(-pupilx)
        # Zernike terms
        for j in range(nz): # creating each row of the matrix
            dzdx = zernike.duZdx(j+1,pupilx,pupily)
            dzdy = zernike.duZdy(j+1,pupilx,pupily)
            norm = np.sqrt(((dzdx*dzdx) + (dzdy*dzdy)).mean())
            if j == 0 and self.dual:
                # dzdx will have shape of pupilx
                # dzdy will have shape of pupilx
                duplicate_dzdx=[]
                duplicate_dzdy=[]
                for i, row in enumerate(mirror2_spots_to_use):
                    for k, spot in enumerate(row):
                        if spot == 1:
                             # spot is on mirror2, so the value should be copied to the duplicate row
                            duplicate_dzdx.append(dzdx[k])
                            duplicate_dzdy.append(dzdy[k])
                        else:
                            # spot is on mirror1, so the value in the duplicate row should be zero
                            duplicate_dzdx.append(0)
                            duplicate_dzdy.append(0)
                            
                dupdzdx = np.reshape(duplicate_dzdx, (nsp,nsp))   
                dupdzdy = np.reshape(duplicate_dzdy, (nsp,nsp))
                duplicate_dzdx = dupdzdx[spots_to_use]
                duplicate_dzdy = dupdzdy[spots_to_use]
                
                zerndx.append(dzdx/norm)
                zerndx.append(duplicate_dzdx/norm)
                zerndy.append(dzdy/norm)
                zerndy.append(duplicate_dzdy/norm)
                
            elif j == 1 and self.dual:
                 # dzdx will have shape of pupilx
                # dzdy will have shape of pupilx
                duplicate_dzdx=[]
                duplicate_dzdy=[]
                for i, row in enumerate(mirror2_spots_to_use):
                    for k, spot in enumerate(row):
                        if spot == 1:
                             # spot is on mirror2, so the value should be copied to the duplicate row
                            duplicate_dzdx.append(dzdx[k])
                            duplicate_dzdy.append(dzdy[k])
                        else:
                            # spot is on mirror1, so the value in the duplicate row should be zero
                            duplicate_dzdx.append(0)
                            duplicate_dzdy.append(0)
                dupdzdx = np.reshape(duplicate_dzdx, (nsp,nsp))   
                dupdzdy = np.reshape(duplicate_dzdy, (nsp,nsp))
                duplicate_dzdx = dupdzdx[spots_to_use]
                duplicate_dzdy = dupdzdy[spots_to_use]
                            
                zerndx.append(dzdx/norm)
                zerndx.append(duplicate_dzdx/norm)
                zerndy.append(dzdy/norm)
                zerndy.append(duplicate_dzdy/norm)
                
            else:
                zerndx.append(dzdx/norm)
                zerndy.append(dzdy/norm)
                norms.append(norm)
                
#         print('zerndx shape', np.array(zerndx).shape)
#         print('zerndy shape', np.array(zerndy).shape)

        zxy = np.hstack([np.array(zerndx),np.array(zerndy)])
    
        #print(zxy)
        self.zxy = zxy
        spots = np.hstack([slopex,slopey]) # the data
        #print(spots)
        zcoeffs=np.linalg.lstsq(zxy.T,spots,rcond=None)[0] # actually perform the fit
        # zernike coefficients evaluated at the locations of the spots
        np.set_printoptions(precision=3,suppress=True,linewidth=200)

        fit = zxy.T.dot(zcoeffs)

        resid = (spots - fit).reshape((2,-1))
        residx = resid[0]
        residy = resid[1]
        rms = resid.std() * np.sqrt(2) * self.pixscale
        
        if self.dual:
            m = get_slope(x1, x2, y1, y2)
            y_intercept = (-m*x1) + y1
            m1  = np.zeros(np.sum(spots_to_use))
            m2  = np.zeros(np.sum(spots_to_use))

            for i, xcoord in enumerate(xcentroids):
                if ycentroids[i] <= (m*xcoord + y_intercept): # below or on the line
                            # MIRROR1
                        m1[i] = 1
                else: # Above the line
                            # MIRROR2
                        m2[i] = 1
                            
            self.m1 = m1
            self.m2 = m2

        fit_ttf = zxy[:4].T.dot(zcoeffs[:4])
        if self.dual: # avoid the duplicated rows
            fit_ttf = np.array([zxy[0], zxy[1], zxy[3], zxy[5]]).T.dot(np.array([zcoeffs[0], zcoeffs[1], zcoeffs[3], zcoeffs[5]])) 
        resid_ttf = (spots - fit_ttf).reshape((2,-1))
        residx_ttf = resid_ttf[0]
        residy_ttf = resid_ttf[1]
        rms_ttf = np.sqrt(residx_ttf.std()**2 + residy_ttf.std()**2) * self.pixscale
        
        self.xcentroids = xcentroids
        self.ycentroids = ycentroids
        self.zcoeffs = zcoeffs

        self.rms = rms
        self.rms_ttf = rms_ttf

        self.xyrefs = np.stack([xrefs, yrefs])
        
        # Tip/tilt/focus residuals
        self.xyrefs_ttf = self.xyrefs + fit_ttf.reshape((2,-1))
        self.xrefs_ttf = self.xyrefs_ttf[0]
        self.yrefs_ttf = self.xyrefs_ttf[1]
        self.xvec = xcentroids - self.xrefs_ttf
        self.yvec = ycentroids - self.yrefs_ttf
        
        # High order xc
        self.xyrefs_ho = self.xyrefs + fit.reshape((2,-1))
        self.xrefs_ho = self.xyrefs_ho[0]
        self.yrefs_ho = self.xyrefs_ho[1]
        self.xvec_ho = xcentroids - self.xrefs_ho
        self.yvec_ho = ycentroids - self.yrefs_ho
        self.norms = np.asarray(norms)
    

    def generateimage(self,n,defocus,zmin,zmax):
        
        # make pupil grid
        py,px = (np.indices((n,n)) - n/2.) / (n/2.)
        pr = np.sqrt(py*py+px*px)
        inpupil = (pr < 1) * (pr>self.obscureratio)
        pupilx = px[inpupil]
        pupily = py[inpupil]
        slopex = np.zeros(pupilx.shape)
        slopey = np.zeros(pupily.shape)
        #for j in range(4,len(sh.zcoeffs)+1):
        for j in range(zmin,zmax+1):
            slopex += zernike.duZdx(j,px[inpupil],py[inpupil]) * self.zcoeffs[j] / self.norms[j] * self.pixscale
            slopey += zernike.duZdy(j,px[inpupil],py[inpupil]) * self.zcoeffs[j] / self.norms[j] * self.pixscale
        slopex += defocus * pupilx
        slopey += defocus * pupily
        pl.figure()
        pl.plot(slopex,slopey,'.')
        pl.axes().set_aspect('equal', 'datalim')

