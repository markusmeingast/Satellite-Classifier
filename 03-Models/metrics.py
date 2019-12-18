"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import glob
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mp

################################################################################
# %% LOAD MODEL
################################################################################

model = load_model('keras_model_roof.h5')

################################################################################
# %% LOAD DATA
################################################################################

for file in glob.glob('../00-raw-data/roof-dataset/pre/image-val*.npy'):
    print(file)
    if 'X_test' not in locals():
        X_test = np.load(file)
        y_test = np.load(file.replace('image','label'))
    else:
        X_test = np.append(X_test, np.load(file), axis=0)
        y_test = np.append(y_test, np.load(file.replace('image','label')), axis=0)

################################################################################
# %% PREDICT!
################################################################################

y_pred = model.predict(X_test[300:371]/255)

y_pred = (y_pred > 0.01).astype(int)

mp.imshow(y_test[300,:,:,0])
mp.imshow(X_test[300,:,:,:])

################################################################################
# %% RUN METRICS
################################################################################

confusion_matrix(y_test[300:371].flatten(),y_pred.flatten())


##### CONVERT DIMENSIONS FOR CONV2D
#X_train = X_train/255
