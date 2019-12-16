import numpy as np
import matplotlib.pyplot as mp

data_img = np.load('image-train-0.npy')
data_lbl = np.load('label-train-0.npy')


mp.imshow(data_img[60])
mp.imshow(data_lbl[60,:,:,0], alpha=0.5)
mp.axis('off')
mp.show()
