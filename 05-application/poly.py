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
from edgetpu.basic.basic_engine import BasicEngine
import numpy as np
import cv2
from mss import mss
from PIL import Image

################################################################################
# %% LOAD MODEL
################################################################################

engine = BasicEngine('../04-deployment/cars_edgetpu.tflite')
(_, xdim, ydim, zdim) = engine.get_input_tensor_shape()

################################################################################
# %% DEFINE SCREEN POSITION
################################################################################

mon = {'top': 200, 'left': 200, 'width': 512, 'height': 512}
sct = mss()

################################################################################
# %% LOOP UNTIL CLOSED
################################################################################

while True:

    ##### GRAB SCREEN
    img = np.array(sct.grab(mon)) #( blue / green / red / alpha? )

    ##### DROP ALPHA
    img = img[:,:,0:3]

    ##### RESIZE TO FIT IN MODEL
    img = cv2.resize(img, (xdim, ydim))

    ##### BGR TO RGB
    img = np.flip(img, axis=2)

    ##### FLATTEN INPUT (TPU REQUIREMENT)
    input = img.flatten()

    ##### RUN ON TPU
    results = engine.run_inference(input)

    ##### RGB TO BGR
    img = np.flip(img, axis=2)

    ##### BINARY
    results = results[1] > 0.5

    ##### REFORMAT RESULTS
    results = (results.reshape(xdim, ydim)*255).astype(np.uint8)

    ##### GET MASK
    mask = results > 128

    #####
    try:
        shapes = features.shapes(results, mask=mask, connectivity=4)
        shapes = as_shapely(shapes)
        shapes = simplify_dp(shapes, 4)
        for i in shapes:
            shape = np.array(i[0].exterior.coords)
            shape = shape.astype(int)
            shape = shape.reshape((-1,1,2))
            if len(shape)>4:
                cv2.fillPoly(img,[shape],(0,255,255,0.2))
                cv2.polylines(img,[shape],True,(0,0,255),1)

    except:
        pass

    ##### RESHAPE FOR PLOTTING
    out = cv2.resize(img, (512, 512))




    cv2.imshow('test', out)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break



"""
################################################################################
# %% LOAD DATA
################################################################################

#X_test = np.load('../00-datasets/roof-dataset/pre/image-val-0.npy')
#y_test = np.load('../00-datasets/roof-dataset/pre/label-val-0.npy')



################################################################################
# %% PLOTTING BASE TRUTH
################################################################################

fig = mp.figure(figsize=(8,8))

##### ID FOR PLOTTING
id = 9

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
    mp.fill(shape[0], shape[1], alpha=0.5)
    mp.plot(shape[0], shape[1], lw=2, color='yellow')

mp.axis([0, 256, 256, 0])
mp.axis('off')
mp.show()
"""
