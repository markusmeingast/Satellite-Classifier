"""
Script to read in trainig image and label data and save into chunks for model training.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import rasterio
import rasterio.plot
import matplotlib.pyplot as mp
import glob
import numpy as np
import sys

################################################################################
# %% DEF FUNCTIONS / CLASS / ETC
################################################################################

def chunk_gen(numbers, n_sec):
    n_sec = max(1, n_sec)
    chunk = [numbers[i:i+n_sec] for i in range(0, len(numbers), n_sec)]
    return chunk

################################################################################
# %% LOAD IMAGES
################################################################################

##### GET IMAGE NUMBERS
numbers = []
for sector in glob.glob('train/image/christchurch_*.tif'):
    numbers.append(sector.split('_')[1][:-4])

##### GET CHUNKS OF 100 FROM NUMBERS
numbers = chunk_gen(numbers,60)

##### ITERATE OVER FILES CHUNKS
for i, chunk in enumerate(numbers):
    print(f'Processing chunk {i}')

    ##### INIT EMPTY NUMPY ARRAY FOR IMAGES
    X = np.zeros((5*5*len(chunk), 500, 500, 3), dtype=np.uint8)
    y = np.zeros((5*5*len(chunk), 500, 500, 1), dtype=np.uint8)

    ##### ITERATE OVER IMAGE NUMBERS (COARSEN FROM 7.5CM TO 30CM RESOLUTION)
    sample_id = 0
    failed = 0
    for number in chunk:
        print(f'Processing --christchurch_{number}.tif--')

        ##### OPEN IMAGE AND LABEL FILES
        image = rasterio.open(f'train/image/christchurch_{number}.tif')
        label = rasterio.open(f'train/label/christchurch_{number}.tif')

        ##### CHECK IF PICTURE SHAPED CORRECTLY
        if image.shape == (10000,10000):
            img = image.read().transpose(1,2,0)[::4,::4,:]
            lbl = label.read().transpose(1,2,0)[::4,::4,:]

            ##### SPLIT INTO 500x500 CHUNKS
            for ix in range(0,2500,500):
                for iy in range(0,2500,500):
                    X[sample_id,0:500,0:500,0:3] = img[iy:iy+500,ix:ix+500,0:3]
                    y[sample_id,0:500,0:500,0:3] = lbl[iy:iy+500,ix:ix+500,0:3]
                    sample_id += 1

        ##### DISCARD IF IMAGE NOT SHAPED CORRECTLY
        else:
            failed += 1

    ##### CORRECT FOR FAILED READS
    if failed > 1:
        X = X[:-failed*5*5,:,:,:]
        y = y[:-failed*5*5,:,:,:]

    ##### SAVE CHUNK TO FILE
    np.save(f'pre/image-{i}.npy', X)
    np.save(f'pre/label-{i}.npy', y)

    break

################################################################################
# %% PLOT
################################################################################
id = 12

mp.imshow(X[id,:,:,:])
mp.imshow(y[id,:,:,0], alpha=0.5)

y.shape
