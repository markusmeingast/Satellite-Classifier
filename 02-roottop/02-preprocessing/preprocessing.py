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
# %% PARAMETERS
################################################################################

case = 'train' # / "train"

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
for sector in glob.glob(f'{case}/image/christchurch_*.tif'):
    numbers.append(sector.split('_')[1][:-4])

##### GET CHUNKS OF 100 FROM NUMBERS
numbers = chunk_gen(numbers, 60)

##### ITERATE OVER FILES CHUNKS
for i, chunk in enumerate(numbers):
    print(f'Processing chunk {i} of {len(numbers)}')


    ##### GLOBAL IMAGE PARAMETERS
    step = 3
    dimx = 256
    dimy = 256
    piecesx = int(np.floor(10000/step/dimx))
    piecesy = int(np.floor(10000/step/dimy))


    ##### INIT EMPTY NUMPY ARRAY FOR IMAGES
    X = np.zeros((piecesx*piecesy*len(chunk), dimx, dimy, 3), dtype=np.uint8)
    y = np.zeros((piecesx*piecesy*len(chunk), dimx, dimy, 1), dtype=np.uint8)

    ##### ITERATE OVER IMAGE NUMBERS (COARSEN FROM 7.5CM TO 30CM RESOLUTION)
    sample_id = 0
    failed = 0
    for number in chunk:
        print(f'Processing --christchurch_{number}.tif--')

        ##### OPEN IMAGE AND LABEL FILES
        image = rasterio.open(f'{case}/image/christchurch_{number}.tif')
        label = rasterio.open(f'{case}/label/christchurch_{number}.tif')

        ##### CHECK IF PICTURE SHAPED CORRECTLY
        if image.shape == (10000, 10000):
            img = image.read().transpose(1,2,0)[::step, ::step,:]
            lbl = label.read().transpose(1,2,0)[::step, ::step,:]

            ##### SPLIT INTO DIMXxDIMY CHUNKS
            for ix in range(0, piecesx*dimx, dimx):
                for iy in range(0, piecesy*dimy, dimy):
                    X[sample_id, 0:dimy, 0:dimx, 0:3] = img[iy:iy+dimy, ix:ix+dimx, 0:3]
                    y[sample_id, 0:dimy, 0:dimx, 0] = lbl[iy:iy+dimy, ix:ix+dimx, 0]
                    sample_id += 1

        ##### DISCARD IF IMAGE NOT SHAPED CORRECTLY
        else:
            failed += 1

    ##### CORRECT FOR FAILED READS
    if failed > 1:
        X = X[:-failed*piecesx*piecesy, :, :, :]
        y = y[:-failed*piecesx*piecesy, :, :, :]

    ##### SAVE CHUNK TO FILE
    np.save(f'pre/image-{case}-{i}.npy', X)
    np.save(f'pre/label-{case}-{i}.npy', y)
