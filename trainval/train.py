"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import sys
from time import time
import matplotlib.pyplot as mp

################################################################################
# %% LOAD DATA AND SPLIT
################################################################################

X = np.load('pre/image-0.npy')
y = np.load('pre/label-0.npy')
X_train, X_test, y_train, y_test = train_test_split(X,y)

################################################################################
# %% NORMALIZE INPUT DATA
################################################################################

X_train = X_train/255.0
X_test = X_test/255.0

################################################################################
# %% BUILD MODEL
################################################################################

##### GET INPUT SHAPE
(_, xdim, ydim, zdim) = X_train.shape

##### ADD LAYERS
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(xdim,ydim,zdim), dtype=tf.float32))
model.add(keras.layers.Conv2D(4, (3, 3), padding='same'))
model.add(keras.layers.Conv2D(10, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same'))

##### COMPILE MODEL AND PRINT SUMMARY
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

##### PRINT SUMMARY
print(model.summary())

################################################################################
# %% INIT CALLBACKS
################################################################################

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), update_freq='epoch')
earlystopping = EarlyStopping(monitor='val_loss', patience=3)

################################################################################
# %% RUN MODEL
################################################################################

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=1,
    verbose=1,
    validation_data=(X_test, y_test),
    use_multiprocessing=True,
    #batch_size=batch_size,
    callbacks=[tensorboard, earlystopping]
)

################################################################################
# %% PLOTTING
################################################################################

id = 16

mp.imshow(X_train[id,:,:,:])
mp.imshow(y_train[id,:,:,0], alpha=0.5)
mp.show()

################################################################################
# %% PLOTTING
################################################################################

X_train.dtype

X_pred = X_train[id]
X_pred = X_pred[np.newaxis]
y_pred = model.predict(X_pred)



mp.imshow(X_pred[0])
mp.imshow(y_pred[0,:,:,0].round(), alpha=0.9)
mp.show()

y_pred.max()
