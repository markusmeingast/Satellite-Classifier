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

""" ROOF
for sector in glob.glob(f'{case}/image/christchurch_*.tif'):
    numbers.append(sector.split('_')[1][:-4])
"""

##### GET IMAGE NUMBERS
numbers = []
for sector in glob.glob(f'../00-datasets/potsdam-dataset/{case}/top_potsdam_*RGB.tif'):
    numbers.append(sector.split('m')[2][:-7])

##### GET CHUNKS OF 100 FROM NUMBERS
numbers = chunk_gen(numbers, 18)

##### ITERATE OVER FILES CHUNKS
for i, chunk in enumerate(numbers):
    if i==2:
        break
    print(f'Processing chunk {i} of {len(numbers)}')

    ##### GLOBAL IMAGE PARAMETERS
    step = 4
    dimx = 256
    dimy = 256
    piecesx = int(np.floor(6000/step/dimx)) # 10000 for AIRS
    piecesy = int(np.floor(6000/step/dimy)) # 10000 for AIRS


    ##### INIT EMPTY NUMPY ARRAY FOR IMAGES
    X = np.zeros((piecesx*piecesy*len(chunk), dimx, dimy, 3), dtype=np.uint8)
    y = np.zeros((piecesx*piecesy*len(chunk), dimx, dimy, 1), dtype=np.uint8)

    ##### ITERATE OVER IMAGE NUMBERS (COARSEN FROM 7.5CM TO 30CM RESOLUTION)
    sample_id = 0
    failed = 0
    for number in chunk:
        print(f'Processing {number} --')

        ##### OPEN IMAGE AND LABEL FILES
        image = rasterio.open(f'../00-datasets/potsdam-dataset/{case}/top_potsdam{number}RGB.tif')
        label = rasterio.open(f'../00-datasets/potsdam-dataset/{case}/top_potsdam{number}label_flat.tif')

        ##### CHECK IF PICTURE SHAPED CORRECTLY
        if image.shape == (6000, 6000):
            img = image.read().transpose(1,2,0)[::step, ::step,:]
            lbl = label.read().transpose(1,2,0)[::step, ::step,:]

            ##### EXTRACT CARS ONLY
            boolr = lbl[:,:,0] == 120
            boolg = lbl[:,:,1] == 120
            boolb = lbl[:,:,2] == 120
            lbl = (boolr*boolg*boolb).astype(int)
            lbl = lbl[:, :, np.newaxis]

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
    np.save(f'../00-datasets/potsdam-dataset/pre/image-{case}-{i}.npy', X)
    np.save(f'../00-datasets/potsdam-dataset/pre/label-{case}-{i}.npy', y)

X.shape
