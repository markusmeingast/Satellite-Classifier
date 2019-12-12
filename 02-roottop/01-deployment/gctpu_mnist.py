"""
This a simple proof of concept script that reads in a EdgeTPU compiled TFLite
model and uses CV2 based webcam capture for inference on a MNIST model.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from edgetpu.basic.basic_engine import BasicEngine
import time
import numpy as np
import cv2
import matplotlib.pyplot as mp

################################################################################
# %% DEFAULT SETTINGS
################################################################################

np.set_printoptions(precision=3)

################################################################################
# %% IMPORT MODEL AND MOVE TO TPU
################################################################################

engine = BasicEngine('mnist_edgetpu.tflite')
(_, xdim, ydim, zdim) = engine.get_input_tensor_shape()

################################################################################
# %% INIT SCREEN OUTPUT
################################################################################

cap = cv2.VideoCapture(0)
mp.figure()

################################################################################
# %% RUN MODEL OFF OF CAMERA ON TPU
################################################################################

while cap.isOpened():

    ##### GRAB IMAGE FROM CAM
    ret, frame = cap.read()
    if not ret:
        break
    image = frame

    ##### RESIZE TO INPUT TENSOR SHAPE
    image = cv2.resize(image, (xdim, ydim))

    ##### CONVERT TO GRAYSCALE
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ##### INVERT
    #image = cv2.bitwise_not(image)

    ##### CONVERT TO BINARY (OTHER OPTIONS MAY MAKE MORE SENSE)
    #_, image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)

    ##### FLATTEN INPUT (TPU REQUIREMENT)
    input = image.flatten()

    ##### RUN ON TPU
    results = engine.run_inference(input)

    ##### REFORMAT RESULTS
    results = (results[1].reshape(xdim, ydim)*255).astype(np.uint8)

    ##### CONVERT TO BINARY (OTHER OPTIONS MAY MAKE MORE SENSE)
    #_, results = cv2.threshold(results,128,255,cv2.THRESH_BINARY)

    ##### PLOT RESULTS
    #mp.gca().cla()
    #mp.bar(np.arange(10),engine.get_raw_output())
    #mp.axis([-0.5,9.5,0,1])
    #mp.pause(0.001)

    ##### SHOW IMAGE THAT WAS FORWARDED TO TPU MODEL
    #image = cv2.resize(image, (560, 560))



    cv2.imshow('frame', results)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

##### CLEAR CAPTURE AND DISPAY
cap.release()
cv2.destroyAllWindows()
