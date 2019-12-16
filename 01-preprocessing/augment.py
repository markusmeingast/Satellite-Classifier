"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import argparse
import glob
import matplotlib.pyplot as mp

################################################################################
# %% SET UP CASE INFO BY COMMAND LINE ARGS
################################################################################

parser = argparse.ArgumentParser(prog='preprocessing.py', usage='%(prog)s <case> <path>', description='Preprocess TIF image data for training')
parser.add_argument('case', help="'train' or 'val'")
parser.add_argument('path', help="path to base dataset")
args = parser.parse_args()

################################################################################
# %% LOAD DATA FILE
################################################################################

##### GET IMAGE NUMBERS
numbers = []
for sector in glob.glob(f'{args.path}/pre/image-{args.case}*.npy'):
    numbers.append(sector.split('-')[-1][:-4])

##### OPEN IMAGES
for number in numbers:
    img = np.load(f'{args.path}/pre/image-{args.case}-{number}.npy')
    lbl = np.load(f'{args.path}/pre/label-{args.case}-{number}.npy')

    dimx, dimy, dimz, dimw = img.shape
    img_out = np.zeros((8*dimx, dimy, dimz, dimw), dtype=img.dtype)
    lbl_out = np.zeros((8*dimx, dimy, dimz, 1), dtype=lbl.dtype)

    idx = 0
    samples = range(len(img))
    for sample in samples:
        ##### SET ORIGINAL
        img_out[idx] = img[sample]
        lbl_out[idx] = lbl[sample]
        idx += 1

        ##### TRANSPOSE
        img_aug = img[sample].transpose(1,0,2)
        img_out[idx] = img_aug

        lbl_aug = lbl[sample].transpose(1,0,2)
        lbl_out[idx] = lbl_aug
        idx += 1

        ##### REPEAT
        for _ in range(3):
            ##### FLIP
            img_aug = np.flip(img_aug, axis=0)
            img_out[idx] = img_aug

            lbl_aug = np.flip(lbl_aug, axis=0)
            lbl_out[idx] = lbl_aug
            idx += 1

            ##### TRANSPOSE
            img_aug = img_aug.transpose(1,0,2)
            img_out[idx] = img_aug

            lbl_aug = lbl_aug.transpose(1,0,2)
            lbl_out[idx] = lbl_aug
            idx += 1

    ##### OUTPUT AUGMENTED FILES
    print(f'Saving to file {args.path}/pre/image-{args.case}-{number}-aug.npy')
    np.save(f'{args.path}/pre/image-{args.case}-{number}-aug.npy', img_out)
    np.save(f'{args.path}/pre/label-{args.case}-{number}-aug.npy', lbl_out)
