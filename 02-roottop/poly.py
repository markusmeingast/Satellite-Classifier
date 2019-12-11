"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from tensorflow import keras
import matplotlib.pyplot as mp
import numpy as np
from rasterio import features
from rastachimp import as_shapely, simplify_dp

################################################################################
# %% LOAD MODEL
################################################################################

model = keras.models.load_model('keras_model.h5')

################################################################################
# %% LOAD DATA
################################################################################

X_test = np.load('pre/image-val-0.npy')
y_test = np.load('pre/label-val-0.npy')

################################################################################
# %% PLOTTING BASE TRUTH
################################################################################

fig = mp.figure(figsize=(8,8))

##### ID FOR PLOTTING
id = 16

mp.imshow(X_test[id,:,:,:])
mp.imshow(y_test[id,:,:,0], alpha=0.5)
mp.axis('off')
mp.show()

################################################################################
# %% PLOTTING PREDICTION ON VALIDATION
################################################################################

fig = mp.figure(figsize=(8,8))

X_pred = X_test[id]/255
X_pred = X_pred[np.newaxis]
y_pred = model.predict(X_pred)

mp.imshow(X_pred[0])
mp.imshow(y_pred[0, :, :, 0]>0.5, alpha=0.5)
mp.axis('off')
mp.show()

################################################################################
# %% SEGMENT AND POLYGONIZE EXAMPLE
################################################################################

fig = mp.figure(figsize=(8,8))

image = y_pred[0,:,:,0].round()
mask = image > 0.5

mp.imshow(X_pred[0])
#mp.imshow(y_pred[0, :, :, 0]>0.5, alpha=0.5)


shapes = features.shapes(image, mask=mask, connectivity=4)
shapes = as_shapely(shapes)
shapes = simplify_dp(shapes, 3)

for i in shapes:
    shape = list(zip(*list(i[0].exterior.coords)))
    mp.fill(shape[0], shape[1], lw=3, alpha=0.5)

mp.axis([0, 256, 256, 0])
mp.axis('off')
mp.show()
