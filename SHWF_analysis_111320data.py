import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom, rotate
import glob
import matplotlib.pyplot as plt
from shanalyzefull import shgridcenter
from sh_analysisfxns_pb import raw_to_npy, get_spots

shwfdir = '/home/pbenac/home/Thesis/raw_SHWF_images'
filelist = glob.glob(shwfdir + '/SHWF-11132020*.raw')
filelist.sort()

[data, images] = raw_to_npy(filelist)


############################
# Images 1 and 2
#############################
# Mirror 2 rotated ~90degrees
spots1 = get_spots(images[1])
spots2 = get_spots(images[2])

y1,x1 = zip(*spots1)
y2,x2 = zip(*spots2)
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.title('Image 1')
plt.imshow(images[1])
plt.scatter(x1,y1)

plt.subplot(122)
plt.title('Image 2')
plt.imshow(images[2])
plt.scatter(x2,y2)
plt.show()


x1 = np.array(x1)
x2 = np.array(x2)
y1 = np.array(y1)
y2 = np.array(y2)
idx1 = np.where(x1>150)[0]
idx2 = np.where(x2>150)[0]
x1_crop = x1[idx1]
x2_crop = x2[idx2]
y1_crop = y1[idx1]
y2_crop = y2[idx2]

d12 = []
for i in range(len(y1_crop)):
        xd = x1_crop[i] - x2_crop[i]
        yd = y1_crop[i] - y2_crop[i]
        d12.append(np.sqrt(xd**2 + yd**2))
        
############################
# Images 3 and 4
#############################
# Mirror 2 rotated ~180degrees
spots3 = get_spots(images[3], pix_apart=15)
spots4 = get_spots(images[4], pix_apart=15)

y3,x3 = zip(*spots3)
y4,x4 = zip(*spots4)
plt.figure(figsize=(12,12))
plt.imshow(images[3])
plt.scatter(x3,y3)
plt.title('Images 3 and 4')

plt.imshow(images[4])
plt.scatter(x4,y4)
plt.show()

x3 = np.array(x3)
x4 = np.array(x4)
y3 = np.array(y3)
y4 = np.array(y4)
idx3 = np.where(x3>150)[0]
idx4 = np.where(x4>150)[0]
x3_crop = x3[idx3]
x4_crop = x4[idx4]
y3_crop = y3[idx3]
y4_crop = y4[idx4]

d34 = []
for i in range(len(y3_crop)):
        xd = x3_crop[i] - x4_crop[i]
        yd = y3_crop[i] - y4_crop[i]
        d34.append(np.sqrt(xd**2 + yd**2))

        

############################
# Images 6 and 7
#############################
# Mirror 1 rotated ~180degrees

spots6 = get_spots(images[6], pix_apart=15)
spots7 = get_spots(images[7], pix_apart=15)

y6,x6 = zip(*spots6)
y7,x7 = zip(*spots7)
plt.figure(figsize=(12,12))
plt.imshow(images[6])
plt.scatter(x6,y6)

plt.imshow(images[7])
plt.title('Images 6 and 7')
plt.scatter(x7,y7)
plt.show()

x6 = np.array(x6)
x7 = np.array(x7)
y6 = np.array(y6)
y7 = np.array(y7)
idx6 = np.where(x6>150)[0]
idx7 = np.where(x7>150)[0]
x6_crop = x6[idx6]
x7_crop = x7[idx7]
y6_crop = y6[idx6]
y7_crop = y7[idx7]

d67 = []
for i in range(len(y7_crop)):
        xd = x6_crop[i] - x7_crop[i]
        yd = y6_crop[i] - y7_crop[i]
        d67.append(np.sqrt(xd**2 + yd**2))