#!/usr/bin/python3

import numpy as np
import pylab as pl
import astropy.io.fits as fits
from matplotlib import animation
from astropy.io import ascii
#%%
def fitbackground(im, backgroundimages, radius):
    yy,xx = np.indices((im.shape))
    yy -= im.shape[0]//2
    xx -= im.shape[1]//2
    rr = np.sqrt(xx*xx+yy*yy)
    outer = (rr>radius).flatten()
    bg = backgroundimages.reshape(backgroundimages.shape[0], -1)
    A = bg[:,outer]
    print(A.shape)
    print(bg.shape)
    fit = np.linalg.lstsq(A,im.flatten()[outer])
    print(fit)

#fitbackground(im,backgroundimages,190)
#%%
# returns xoffset, yoffset
def shgridcenter(shimage, gridspacing=14):
    #gridspacing = 14
    offset = np.zeros((2))
    for axis in [0,1]:
        cut = shimage.sum(axis=axis)
        center = len(cut) // 2
        offset[axis] =  np.argmax(cut[center-gridspacing//2:center+gridspacing//2]) - gridspacing//2
    return (offset)

 #%%   
# Find the edges of the pupil assuming we have a central obscuration
# by locating the places where the intensity crosses the 50% value
#
def computecenter(pslicey):
    #center of pupil image
    psliceymax = np.amax(pslicey)
    dp = pslicey[1:] - pslicey[:-1]
    dx = ( psliceymax/2 - pslicey[1:] ) / dp
    crossings = abs(dx+0.5) < 0.5 
    yvals = (dx+range(1,nx))[crossings]

    pupildiay = yvals[3] - yvals[0]
    pupilobsc = yvals[2] - yvals[1]
    pupilceny1 = (yvals[3] + yvals[0]) / 2.
    pupilceny2 = (yvals[1] + yvals[2]) / 2.
    print(pupildiay, pupilceny1, pupilceny2)

    return(pupildiay, pupilobsc, (pupilceny1+pupilceny2)/2.)



def makecube(listfilename, stackfilename):
    # not used, takes individual images and makes cube
    datadir = '/data/agws/wfs_images/'
    skyfile = datadir + 'mg2201811180604590157.fits'

    skyimg = fits.open(skyfile)[0].data
    imgstack = []
    timestamps = []
    t = ascii.read(listfilename)
    for fitsname in t['wfsname']:
        hdu = fits.open(datadir+fitsname.strip())[0]
        img = hdu.data.astype(float)
        hdr = hdu.header
        timestamps.append(hdr['DATE-OBS']+'T'+hdr['UT'])
        imgstack.append(img-skyimg)

    imgstack = np.array(imgstack)
            
    fits.writeto(stackfilename,imgstack,overwrite=True)
    return(timestamps)

#%%
if __name__ == "__main__": 
#     listfilename = '/home/bmcleod/analysis/proto3c/matched_files.txt'
#     listfilename = '/home/bmcleod/analysis/proto3c/onefile.txt'
#     stackfilename = '/home/bmcleod/analysis/wfsimages/imgstack.fits'

    timestamps = makecube(listfilename,stackfilename)

    #%%
    cube = fits.open(stackfilename)[0].data
    #%%
    # Rotation angle between SH and Proto3 coordinates
    rotangle = np.radians(27.8)

    colorcodes = ['black', '#cd5c5c', '#4682b4'] 

    pixscale = 0.287 # asec per pix
    spacing = 6.5 / 30 # meters
    dx = 320 / 28.
    dy = 320 / 28.
    x0 = int (254 - 320/2 - 2.5 * dx)
    y0 = int (262 - 320/2 - 2.5 * dy)
    nx = 34
    ny = 34
    nframes = len(cube)

    #led refererence frame
#     templateimagename = '/data/agws/wfs_images/mg2201811162322310838.fits'

    # Process the template image, find and centroid spots
    templateimage = fits.open(templateimagename)[0].data
    xtemplate = np.zeros((nx,ny))
    ytemplate = np.zeros((nx,ny))

    # planning to implement correlation, unused
    ix = np.arange(5) - 2
    gauss1d = np.exp((ix*ix)/2.)

    #loop over spot grid
    for iy in range(ny):
        for ix in range(nx):
            xmin = int(x0 + ix * dx)
            ymin = int(y0 + iy * dy)
            xmax = xmin + int(dx)
            ymax = ymin + int(dy)
            box = templateimage[ymin:ymax,xmin:xmax]

            bkgval = np.median(box)
            box = box - bkgval
            thresh = 2
            yval,xval = np.indices(box.shape)
            counts = box[box>thresh].sum() 
            xcen = (box[box>thresh]*xval[box>thresh]).sum() / counts
            ycen = (box[box>thresh]*yval[box>thresh]).sum() / counts
            xtemplate[iy,ix] = xcen
            ytemplate[iy,ix] = ycen

    #repeat for each data image
    xarr = np.zeros((nframes,ny,nx))
    yarr = np.zeros((nframes,ny,nx))
    countarr   = np.zeros((nframes,ny,nx))
    pistonxarr = np.zeros((nframes))
    pistonyarr = np.zeros((nframes))

    # Loop through the frames
    for iframe in range(nframes):
        # For each subaperture compute the centroid
        for iy in range(ny):
            for ix in range(nx):
                xmin = int(x0 + ix * dx)
                ymin = int(y0 + iy * dy)
                xmax = xmin + int(dx)
                ymax = ymin + int(dy)
                box = cube[iframe][ymin:ymax,xmin:xmax]
                bkgval = np.median(box)
                box = box - bkgval
                thresh = 2
                yval,xval = np.indices(box.shape)
                counts = box[box>thresh].sum()
                if counts>0:
                    xcen = (box[box>thresh]*xval[box>thresh]).sum() / counts
                    ycen = (box[box>thresh]*yval[box>thresh]).sum() / counts
                else:
                    xcen = 0
                    ycen = 0
                xarr[iframe,iy,ix] = xcen - xtemplate[iy,ix]
                yarr[iframe,iy,ix] = ycen - ytemplate[iy,ix]
                countarr[iframe,iy,ix] = counts

    # Find the middle of the pupil, refine based on 
    pupilmap = countarr.mean(axis=0)
    print(pupilmap.shape)

    #center ten rows and columns
    pslicex = pupilmap[int(ny/2-5):int(ny/2+5),:].sum(axis=0)
    pslicey = pupilmap[:,int(nx/2-5):int(nx/2+5)].sum(axis=1)
    #pl.plot(pslicex)
    #pl.plot(pslicey)
    #pl.show()

    pupildiax, pupilobscx, pupilcenx = computecenter(pslicex)
    pupildiay, pupilobscy, pupilceny = computecenter(pslicey)

    # get coords of pupil with radius of one
    py,px = np.indices(pupilmap.shape)
    normalizedpupilx = (px - pupilcenx)/(pupildiax/2.)
    normalizedpupily = (py - pupilceny)/(pupildiay/2.)
    obscuration = ((pupilobscx / pupildiax) + (pupilobscy / pupildiay)) / 2.
    pupilrad = np.sqrt(normalizedpupilx**2 + normalizedpupily**2)
    inpupil =  (pupilrad < 1) * (pupilrad > obscuration)

    iframe = 0
    meancounts = countarr[iframe][inpupil].mean()
    goodsubaps = ((countarr[iframe] > meancounts/3.))


    # Rotate the SH pupil coordinates to Proto3 coordinates and normalize to meters

    rotnormx = (np.cos(rotangle) * normalizedpupilx - np.sin(rotangle) * normalizedpupily)
    rotnormy = (np.sin(rotangle) * normalizedpupilx + np.cos(rotangle) * normalizedpupily)
    rotpx = 3.24 * rotnormx
    rotpy = 3.24 * rotnormy

    # Rotate the centroids

    rotcenx = np.cos(rotangle) * xarr - np.sin(rotangle) * yarr
    rotceny = np.sin(rotangle) * xarr + np.cos(rotangle) * yarr

    fig, ax = pl.subplots(1,1)
    Q = ax.quiver(rotnormx[goodsubaps], rotnormy[goodsubaps], rotcenx[iframe][goodsubaps], rotceny[iframe][goodsubaps])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.set_aspect('equal','datalim')
    Tcount = ax.text(1,  1, "0")
    Trms   = ax.text(1, -1, "0")

def update_quiver(framenum):
    x = rotcenx[framenum%nframes][goodsubaps]
    y = rotceny[framenum%nframes][goodsubaps]
    rms = np.sqrt(x*x + y*y).std() * pixscale
    rms_str = "%.2f" % (rms)
    Q.set_UVC(x - x.mean(), y - y.mean())
    Tcount.set_text(str(framenum%nframes))
    Trms.set_text(rms_str)
    return Q


if __name__ == "__main__": 
    # Phasing aperture locations in polar coordinates
    aper_rad = 2.073 # meters

    aper_theta = np.radians(np.array([270., 150., 30.]))
    aper_x = aper_rad * np.cos(aper_theta)
    aper_y = aper_rad * np.sin(aper_theta)

    for ax, ay in zip(aper_x, aper_y):
        cornerx = [-0.75, -0.75, 0.75, 0.75, -0.75]
        cornery = [-0.75, 0.75,  0.75, -0.75, -0.75]
        pl.plot((cornerx + ax), (cornery + ay))

    #anim = animation.FuncAnimation(fig, update_quiver, interval=nframes, blit=False)
    #fig.tight_layout()              


    # Make image of which phasing subaps are in use for calcs
    # Compute piston
    apernum = 1
    pl.figure()
    pistonvals = []
    pl.plot(normalizedpupilx[countarr[0]>50], normalizedpupily[countarr[0]>50],'b.')
    for ax, ay in zip(aper_x, aper_y):
        illum = (abs(rotpx - ax) < 0.75) * (abs(rotpx - ax) > 0.25) * (abs(rotpy - ay) < 0.75)
        ingap = (abs(rotpx - ax) < 0.25) * (abs(rotpy - ay) < 0.75)
        pl.plot(normalizedpupilx[illum], normalizedpupily[illum],'ro')
        pl.plot(normalizedpupilx[ingap], normalizedpupily[ingap],'go')

        #compute piston difference
        gapsize = 0.5 # meters
        for iframe in range(nframes):
            xtilt = xarr[iframe][illum].mean()
            pistonxarr[iframe] = (xarr[iframe][ingap].mean() - xtilt)  * pixscale / 206265 * gapsize * 1e9

        pistonvals.append(pistonxarr.copy())

        apernum += 1
    pl.axes().set_aspect('equal','datalim')

    pl.figure()
    for apernum in range(1,4):
        pl.plot(pistonvals[apernum-1],label = "Aper "+str(apernum), color=colorcodes[apernum-1])
    pl.legend()
    pl.title('SH derived piston')
    pl.ylabel('piston (nm)')
    pl.xlabel('Frame number')          

    piston_file = open("sh_pistons.txt","w")
    piston_file.write("UT\tAP1\t AP2\tAP3\tcounts\n");
    for loop in range(nframes):
        piston_file.write("%s\t%f\t%f\t%f\n%f" % (timestamps[loop],
                                                  pistonvals[0][loop],
                                                  pistonvals[1][loop],
                                                  pistonvals[2][loop],
                                                  meancounts[loop]))
    piston_file.close()

#     fits.writeto("/home/bmcleod/analysis/proto3c/intensity.fits",countarr,overwrite=True)

    pl.show()

#        xtilt = xarr[iframe,:,illum].mean()
#        ytilt = yarr[iframe,illum,:].mean()
#
#        pistonyarr[iframe] = (yarr[iframe,~illum,:].mean() - ytilt)  * pixscale / 206265 * spacing * 2 * 1e9

