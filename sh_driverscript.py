import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom, rotate
import scipy
import glob
import matplotlib.pyplot as plt
from shanalyzefull import shgridcenter
from sh_analysisfxns_pb import raw_to_npy
import zernike
import sh_cleaned as sh

# Replace with appropriate directory
shwfdir = '/home/pbenac/home/Thesis/raw_SHWF_images'
filelist = glob.glob(shwfdir + '/SHWF-11132020*.raw')
filelist.sort()

#load data, images
[data, images] = raw_to_npy(filelist)

img1 = sh.prep_image(images[1], crop_ymin=0, crop_ymax=350)

#Instantiate SH class 
img1_TM = sh.SH(img1, flux_frac=0.2, two_mirrors=True)
img1_TM.shgridcenter(img1)

# Run the least squares fit - nzern=10, and the points on the line are (100,275) and (300,125)
img1_TM.process(10, 100, 275, 300, 125) 

img1_TM.make_twomirror_vectorplot(plot_title='Image 1')

# plt.savefig('image1_twomirror_warrows.png')
img1_TM.get_offsets_twomirrors()
print('img1 tilt difference', img1_TM.total_difference*1e3, 'mrad')
print('img1 xtilt', img1_TM.xTiltDifference*1e3, 'mrad')
print('img1 ytilt', img1_TM.yTiltDifference*1e3, 'mrad')
print('Difference (pixels)', img1_TM.difference_pixels)
# print('xvectors', img1_TM.xvec)
# print('yvectors', img1_TM.yvec)

# IMAGE 2
img2 = sh.prep_image(images[2], crop_ymin=0, crop_ymax=350)
# plt.imshow(img2)
img2_TM = sh.SH(img2, flux_frac=0.2, two_mirrors=True)
img2_TM.shgridcenter(img2)

# Run the least squares fit - nzern=10, and the points on the line are (100,275) and (300,125)
img2_TM.process(10, 100, 275, 300, 125)
img2_TM.get_offsets_twomirrors()
print('img2 tilt difference', img2_TM.total_difference*1e3, 'mrad')
print('img2 xtilt', img2_TM.xTiltDifference*1e3, 'mrad')
print('img2 ytilt', img2_TM.yTiltDifference*1e3, 'mrad')
print('Difference (pixels)', img2_TM.difference_pixels)


img2_TM.make_twomirror_vectorplot(plot_title='Image 2')

# IMAGE 3
img3 = sh.prep_image(images[3], crop_xmax=380, crop_ymin=20, crop_ymax=400)
# plt.imshow(img3)
img3_TM = sh.SH(img3, flux_frac=0.2, two_mirrors=True)
img3_TM.shgridcenter(img3)

# Run the least squares fit - nzern=10, and the points on the line are (100,275) and (300,125)
img3_TM.process(10, 125, 325, 340, 135)
img3_TM.get_offsets_twomirrors()
print('img3 tilt difference', img3_TM.total_difference*1e3, 'mrad')
print('img3 xtilt', img3_TM.xTiltDifference*1e3, 'mrad')
print('img3 ytilt', img3_TM.yTiltDifference*1e3, 'mrad')
print('Difference (pixels)', img3_TM.difference_pixels)

img3_TM.make_twomirror_vectorplot(plot_title='Image 3')

# IMAGE 4
img4 = sh.prep_image(images[4], crop_xmax=380, crop_ymin=20, crop_ymax=400)
img4_TM = sh.SH(img4, flux_frac=0.2, two_mirrors=True)
img4_TM.shgridcenter(img4)

# Run the least squares fit - nzern=10, and the points on the line are (100,275) and (300,125)
img4_TM.process(10, 125, 325, 340, 135)
img4_TM.get_offsets_twomirrors()
print('img4 tilt difference', img4_TM.total_difference*1e3, 'mrad')
print('img4 xtilt', img4_TM.xTiltDifference*1e3, 'mrad')
print('img4 ytilt', img4_TM.yTiltDifference*1e3, 'mrad')
print('Difference (pixels)', img4_TM.difference_pixels)

img4_TM.make_twomirror_vectorplot(plot_title='Image 4')

# # IMAGE 5
# img5 = sh.prep_image(images[5], crop_xmax=380, crop_ymin=20, crop_ymax=400)
# plt.imshow(img5)
# plt.show()
# img5_TM = sh.SH(img5, flux_frac=0.2, two_mirrors=True)
# img5_TM.shgridcenter(img5)

# # Run the least squares fit - nzern=10, and the points on the line are (100,275) and (300,125)
# img5_TM.process(10, 125, 325, 340, 135)
# img5_TM.get_offsets_twomirrors()
# print('img5 tilt difference', img5_TM.total_difference*1e3, 'mrad')
# print('img5 xtilt', img5_TM.xTiltDifference*1e3, 'mrad')
# print('img5 ytilt', img5_TM.yTiltDifference*1e3, 'mrad')

# img5_TM.make_twomirror_vectorplot(plot_title='Image 5')


# Image 6
img6 = sh.prep_image(images[6], crop_xmax=380, crop_ymin=20, crop_ymax=400)
plt.imshow(img6)
plt.show()
img6_TM = sh.SH(img6, flux_frac=0.2, two_mirrors=True)
img6_TM.shgridcenter(img6)

# Run the least squares fit - nzern=10, and the points on the line are (100,275) and (300,125)
img6_TM.process(10, 130, 270, 340, 135)
img6_TM.get_offsets_twomirrors()
print('img6 tilt difference', img6_TM.total_difference*1e3, 'mrad')
print('img6 xtilt', img6_TM.xTiltDifference*1e3, 'mrad')
print('img6 ytilt', img6_TM.yTiltDifference*1e3, 'mrad')
print('Difference (pixels)', img6_TM.difference_pixels)
print('Tilt diff. mrad', sh.offset_dist_to_tilt(img6_TM.difference_pixels * 16 / 1e3, return_mrad=True))
img6_TM.make_twomirror_vectorplot(plot_title='Image 6')