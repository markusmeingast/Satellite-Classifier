"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import rasterio
import rasterio.merge as merge
from rasterio.plot import show, reshape_as_image
import matplotlib.pyplot as mp
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import numpy as np
import colormap
import cv2

################################################################################
# %% BUILD COLOR PALETTE FROM EE DATA
################################################################################

##### CORINE COLOR CODE
palette_rgb = []
palette_hex = [
    '#E6004D', '#FF0000', '#CC4DF2', '#CC0000', '#E6CCCC',
    '#E6CCE6', '#A600CC', '#A64DCC', '#FF4DFF', '#FFA6FF',
    '#FFE6FF', '#FFFFA8', '#FFFF00', '#E6E600', '#E68000',
    '#F2A64D', '#E6A600', '#E6E64D', '#FFE6A6', '#FFE64D',
    '#E6CC4D', '#F2CCA6', '#80FF00', '#00A600', '#4DFF00',
    '#CCF24D', '#A6FF80', '#A6E64D', '#A6F200', '#E6E6E6',
    '#CCCCCC', '#CCFFCC', '#000000', '#A6E6CC', '#A6A6FF',
    '#4D4DFF', '#CCCCFF', '#E6E6FF', '#A6A6E6', '#00CCF2',
    '#80F2E6', '#00FFA6', '#A6FFE6', '#E6F2FF']

##### CONVERT TO MATPLOTLIB COLORMAP
for color in palette_hex:
    palette_rgb.append(colormap.colors.hex2rgb(color))
palette_rgb = tuple(tuple(i / 255.0 for i in inner) for inner in palette_rgb)
cm = LinearSegmentedColormap.from_list('google', palette_rgb)

################################################################################
# %% IMPORT IMAGE/LANDCOVER DATA
################################################################################

city = 'berlin'
image = rasterio.open(f'image-{city}-2015.tif')
landcover = rasterio.open(f'landcover-{city}-2012.tif')

################################################################################
# %% PLOT OVERLAY
################################################################################
"""
fig, ax = mp.subplots(1,1)
show(image, ax=ax, vmin=0.0, vmax=0.3)
show(landcover, alpha=0.3, vmin=1, vmax=len(palette_hex), cmap=cm, ax=ax)
mp.gca().set_aspect(1.0/np.cos(image.lnglat()[1]*np.pi/180.0))
mp.savefig(f'{city}-overlay.png', dpi=1200)
mp.show()
"""
################################################################################
# %% LOOP OVER INTERNAL LANDCOVER PIXELS
################################################################################

##### GET INFO ON BOUNDING BOX FOR LANDCOVER
bbox_lc = landcover.bounds[:]

xmin_lc = bbox_lc[0]
xmax_lc = bbox_lc[2]
ymin_lc = bbox_lc[1]
ymax_lc = bbox_lc[3]

dx_lc = (xmax_lc-xmin_lc)/landcover.shape[1]
dy_lc = (ymax_lc-ymin_lc)/landcover.shape[0]

##### GET INFO ON BOUNDING BOX FOR IMAGE
bbox_im = image.bounds[:]

xmin_im = bbox_im[0]
xmax_im = bbox_im[2]
ymin_im = bbox_im[1]
ymax_im = bbox_im[3]

dx_im = (xmax_im-xmin_im)/image.shape[1]
dy_im = (ymax_im-ymin_im)/image.shape[0]

##### CALCULATE OFFSET OF IMAGES IN PIXELS BASED ON TOP LEFT CORNER
x_offset = 50-int((xmin_im - xmin_lc)/dx_im)
y_offset = 50-int((ymax_lc - ymax_im)/dy_im)

##### CROP LANDCOVER ARRAY
landcover_arr = landcover.read()
landcover_arr = landcover_arr[:,1:-1,1:-1]

##### CROP IMAGE ARRAY
image_arr = image.read()
x_start = x_offset
x_end = np.int(np.floor((image.shape[1]-x_offset)/50))*50+x_offset
y_start = y_offset
y_end = np.int(np.floor((image.shape[0]-y_offset)/50))*50+y_offset
image_arr = image_arr[:,y_start:y_end,x_start:x_end]

##### CONVERT ARRAY TO 50x50 IMAGES
image_arr = image_arr.transpose((1,2,0))
n_samples = int(image_arr.shape[0]*image_arr.shape[1]/50/50)
images = np.zeros((n_samples,50,50,4), dtype=int)

sample_id = 0
for ix in range(0,image_arr.shape[1],50):
    for iy in range(0,image_arr.shape[0],50):
        images[sample_id,0:50,0:50,0:4] = image_arr[iy:iy+50,ix:ix+50,0:4]
        sample_id += 1

categories = landcover_arr.T.flatten()

np.savetxt(f'target-{city}.csv', categories, fmt='%2i')
np.savetxt(f'input-{city}.csv', images)

################################################################################
# %% DUMP
################################################################################


"""
mp.figure()
ax = mp.gca()

##### LOOP OVER INTERNAL PIXELS
cover = [ ]
#for ix in range(1,landcover.shape[1]-1):
#    for iy in range(1,landcover.shape[1]-1):

#for ix in range(1,2):
#    for iy in range(1,2):

#        cover.append(landcover_arr[0,iy,ix])

show(image, ax=ax, vmin=0.0, vmax=0.3)
show(landcover, alpha=0.3, vmin=1, vmax=len(palette_hex), cmap=cm, ax=ax)

##### CALCULATE LOCAL BOUNDS FROM LANDCOVER
xskip = 120
yskip = 118
xmin_local = xmin_lc + (xskip+1)*dx_lc
xmax_local = xmin_lc + (xskip+2)*dx_lc
ymin_local = ymax_lc - (yskip+1)*dy_lc
ymax_local = ymax_lc - (yskip+2)*dy_lc

mp.plot([xmin_local,xmin_local],[ymin_local,ymax_local],'k-')
mp.plot([xmin_local,xmax_local],[ymin_local,ymin_local],'k-')
mp.plot([xmax_local,xmax_local],[ymin_local,ymax_local],'k-')
mp.plot([xmin_local,xmax_local],[ymax_local,ymax_local],'k-')

xmin_local = xmin_im + x_offset*dx_im + (xskip+0)*50*dx_im
xmax_local = xmin_im + x_offset*dx_im + (xskip+1)*50*dx_im
ymin_local = ymax_im - y_offset*dy_im - (yskip+0)*50*dy_im
ymax_local = ymax_im - y_offset*dy_im - (yskip+1)*50*dy_im

mp.plot([xmin_local,xmin_local],[ymin_local,ymax_local],'k-')
mp.plot([xmin_local,xmax_local],[ymin_local,ymin_local],'k-')
mp.plot([xmax_local,xmax_local],[ymin_local,ymax_local],'k-')
mp.plot([xmin_local,xmax_local],[ymax_local,ymax_local],'k-')




mp.show()



################################################################################
# %% DUMP
################################################################################

118*120+120
"""
