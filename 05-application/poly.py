"""
Script that grabs image data from screen to overlay semantic segmentation
results based on that image.
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
import time

################################################################################
# %% STOPWATCH
################################################################################

class StopWatch(object):

    def __init__(self, string):
        self.start_time = None
        self.stop_time = None
        self.string = string

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.stop_time = time.time()
        print(f"{self.string.ljust(30)}: {(self.stop_time-self.start_time):2.3f} s")

################################################################################
# %% LOAD MODEL
################################################################################

engine = BasicEngine('../04-deployment/merged_edgetpu.tflite')
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

    ##### INIT EMPTY OVERLAY AND RESULT MAPS
    ovr = np.zeros((img.shape), dtype=np.uint8)
    res = np.zeros((img.shape), dtype=np.uint8)

    ##### BGR TO RGB
    img = np.flip(img, axis=2)

    ##### FLATTEN INPUT (TPU REQUIREMENT)
    input = img.flatten()

    ##### RUN ON TPU
    results = engine.run_inference(input)

    ##### RGB TO BGR
    img = np.flip(img, axis=2)

    ##### REFORMAT RESULTS
    results = (results[1].reshape(xdim, ydim, 2)*255).astype(np.uint8)

    ##### GET MASK
    output = results[:,:,1]
    mask = output > 128
    res[:,:,1] = output
    output = (output > 128).astype(np.uint8)*255

    #####
    try:
        shapes = features.shapes(output, mask=mask, connectivity=4)
        shapes = as_shapely(shapes)
        shapes = simplify_dp(shapes, 5)
        for i in shapes:
            shape = np.array(i[0].exterior.coords)
            shape = shape.astype(int)
            shape = shape.reshape((-1,1,2))
            if len(shape)>4:
                cv2.fillPoly(ovr,[shape],(0,255,255,0.2))
                cv2.polylines(img,[shape],True,(0,0,255),2)
    except:
        pass

    ##### GET MASK
    output = results[:,:,0]
    mask = output > 128
    res[:,:,0] = output
    output = (output > 128).astype(np.uint8)*255

    #####
    try:
        shapes = features.shapes(output, mask=mask, connectivity=4)
        shapes = as_shapely(shapes)
        shapes = simplify_dp(shapes, 2)
        for i in shapes:
            shape = np.array(i[0].exterior.coords)
            shape = shape.astype(int)
            shape = shape.reshape((-1,1,2))
            if len(shape)>4:
                cv2.fillPoly(ovr,[shape],(255,0,0,0.2))
                cv2.polylines(img,[shape],True,(255,0,255),2)
    except:
        pass

    ##### OVERLAY ORIGINAL IMAGE WITH POLYGONS
    img = cv2.addWeighted(img,1.0,ovr,0.5,0)

    ##### RESHAPE FOR PLOTTING
    out = cv2.resize(img, (512, 512))

    ##### DISPLAY IMAGE AND RESULTS
    cv2.imshow('polygonized', out)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    cv2.imshow('raw', res)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
