"""
Script to read in trainig image and label data and save into chunks for model training.

e.g.:
 - python3 preprocessing.py train ../00-raw-data/roof-dataset 4096
 - python3 preprocessing.py train ../00-raw-data/potsdam-dataset 128
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
import argparse

################################################################################
# %% GET CHUNKS FROM LIST
################################################################################

#def chunk_gen(numbers, n_sec):
#    n_sec = max(1, n_sec)
#    chunk = [numbers[i:i+n_sec] for i in range(0, len(numbers), n_sec)]
#    return chunk

################################################################################
# %% SET UP CASE INFO BY COMMAND LINE ARGS
################################################################################

parser = argparse.ArgumentParser(prog='preprocessing.py', usage='%(prog)s <case> <path> <chunk>', description='Preprocess TIF image data for training')
parser.add_argument('case', help="'train' or 'val'")
parser.add_argument('path', help="path to base dataset")
parser.add_argument('chunk', help="no. of samples per chunk")
parser.add_argument('--color', help="tuple of uint8 color (e.g. (255,255,255))")#  <-- doesn't work yet
args = parser.parse_args()

################################################################################
# %% LOAD IMAGES FOR CASE
################################################################################

##### GET IMAGE NUMBERS
numbers = []
for sector in glob.glob(f'{args.path}/{args.case}/image/*.tif'):
    numbers.append(sector.split('-')[-1][:-4])

##### GLOBAL IMAGE PARAMETERS
step = 1
dimx = 256
dimy = 256

##### INIT EMPTY NUMPY ARRAY FOR IMAGES
X = np.zeros((int(args.chunk), dimx, dimy, 3), dtype=np.uint8)
y = np.zeros((int(args.chunk), dimx, dimy, 1), dtype=np.uint8)

##### ITERATE OVER FILES CHUNKS
out_idx = 0
sample_id = 0
for i, number in enumerate(numbers):
    print(f'Processing image {i+1} of {len(numbers)}: {number}.tif')

    ##### OPEN IMAGE AND LABEL FILES
    img = rasterio.open(glob.glob(f'{args.path}/{args.case}/image/*-{number}.tif')[0]).read().transpose(1,2,0)[::step, ::step,:]
    lbl = rasterio.open(glob.glob(f'{args.path}/{args.case}/label/*-{number}.png')[0]).read().transpose(1,2,0)[::step, ::step,:]

    if lbl.shape[2] > 1:
        if args.color is not None:
            ##### EXTRACT COLOR CODE ONLY (MULTICLASS/POTSDAM)
            boolr = lbl[:,:,0] == 255
            boolg = lbl[:,:,1] == 0
            boolb = lbl[:,:,2] == 0

            lbl = (boolr*boolg*boolb).astype(int)
            lbl = lbl[:, :, np.newaxis]
        else:
            sys.exit('Color not defined or defined correctly!')

    ##### GET IMAGE DIMENSIONS
    pix, piy, _ = img.shape

    ##### GET NUMBER OF SECTIONS FOR IMAGE
    piecesx = int(np.floor(pix/dimx))
    piecesy = int(np.floor(piy/dimy))

    print(piecesx)
    print(piecesy)

    ##### SPLIT IMAGE INTO DIMXxDIMY CHUNKS
    for ix in range(0, piecesx*dimx, dimx):
        for iy in range(0, piecesy*dimy, dimy):
            img_section = img[ix:ix+dimx, iy:iy+dimy, 0:3]
            lbl_section = lbl[ix:ix+dimx, iy:iy+dimy, 0]

            ##### CHECK IF LABEL CONTAINS ANY TARGET
            if args.case == 'train':
                if lbl_section.max() == 1:
                    print(f'Adding sample {sample_id}')

                    ##### SET SAMPLE
                    X[sample_id, 0:dimy, 0:dimx, 0:3] = img_section
                    y[sample_id, 0:dimy, 0:dimx, 0] = lbl_section

                    ##### WRITE TO FILE IF ENOUGH SAMPLES WERE WRITTEN
                    if sample_id == int(args.chunk)-1:
                        ##### SAVE CHUNK TO FILE
                        print(f'Writing file!')
                        np.save(f'{args.path}/pre/image-{args.case}-{out_idx}.npy', X)
                        np.save(f'{args.path}/pre/label-{args.case}-{out_idx}.npy', y)
                        sample_id = 0
                        out_idx += 1
                    else:
                        sample_id += 1
            else:
                print(f'Adding sample {sample_id}')

                ##### SET SAMPLE
                X[sample_id, 0:dimy, 0:dimx, 0:3] = img_section
                y[sample_id, 0:dimy, 0:dimx, 0] = lbl_section

                ##### WRITE TO FILE IF ENOUGH SAMPLES WERE WRITTEN
                if sample_id == int(args.chunk)-1:
                    ##### SAVE CHUNK TO FILE
                    print(f'Writing file!')
                    np.save(f'{args.path}/pre/image-{args.case}-{out_idx}.npy', X)
                    np.save(f'{args.path}/pre/label-{args.case}-{out_idx}.npy', y)
                    sample_id = 0
                    out_idx += 1
                else:
                    sample_id += 1
