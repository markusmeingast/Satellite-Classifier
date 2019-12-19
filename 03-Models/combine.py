"""
Currently TF/Keras has issues exporting the combined model from two different
trained models. One alternative (although fussy) is to "recreate" an merged
untrained version of the model and manually set the weights per layer from the
trained models. This approach is pursued below
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, concatenate
import numpy as np
import matplotlib.pyplot as mp
from tensorflow.keras.utils import plot_model

################################################################################
# %% TRANSFER WEIGHTS
################################################################################

##### LOAD BLANK/UNTRAINED MODEL, CREATED WITH REBUILD.PY
merged = load_model('keras_model_merged_untrained.h5')

##### READ THE TRAINED MODELS
model_cars = load_model('keras_model_cars.h5')
model_roof = load_model('keras_model_roof.h5')
model_road = load_model('keras_model_road.h5')

##### ITERATE OVER THE LAYERS IN THE BLANK MODEL
lvl = 1
for i in range(1,len(merged.layers)-1):
    if i%2 == 1:
        merged.layers[i].set_weights(model_roof.layers[lvl].get_weights())
    else:
        merged.layers[i].set_weights(model_road.layers[lvl].get_weights())
        lvl += 1

##### SAVE COMBINED AND "TRAINED" MODEL
merged.save('keras_model_merged.h5')

################################################################################
# %% LOAD TEST DATA
################################################################################

X_test = np.load('../00-raw-data/potsdam-dataset/pre/image-train-0.npy')
y_test = np.load('../00-raw-data/potsdam-dataset/pre/label-train-0.npy')

################################################################################
# %% PLOT BASE TRUTH
################################################################################

idx= 120
mp.imshow(X_test[idx])
mp.imshow(y_test[idx,:,:,0], alpha=0.5)
mp.show()

################################################################################
# %% PLOT PREDICTIONS
################################################################################

y_pred = merged.predict(X_test[idx][np.newaxis,:,:,:])
y_pred1 = y_pred[0,:,:,0]
y_pred2 = y_pred[0,:,:,1]
y_pred3 = y_pred[0,:,:,2]

mp.imshow(X_test[idx])
mp.imshow(y_pred1, alpha=0.5)
mp.show()

mp.imshow(X_test[idx])
mp.imshow(y_pred2, alpha=0.5)
mp.show()

mp.imshow(X_test[idx])
mp.imshow(y_pred3, alpha=0.5)
mp.show()

################################################################################
# %% THIS APPROACH SHOULD WORK IN THEORY, MODEL PRODUCES CORRECT PREDICTIONS
#    BUT CANNOT BE SAVED TO H5, NEEDED FOR TFLITE
################################################################################

"""

##### BUILD COMMON INPUT LAYER
input_size = (256,256,3)
inputs = Input(input_size)

##### LOAD CARS MODEL
model_cars = load_model('keras_model_cars.h5')
##### RENAME LAYERS TO BE UNIQUE
i = 0
for layer in model_cars.layers:
    #if i==0:
    #    pass
    #else:
        layer._name = layer._name + str("_cars")
##### RENAME MODEL TO BE UNIQUE
model_cars._name = 'model_cars'
##### DROP INPUT LAYER
#model_cars._layers.pop(0)

##### LOAD ROOF MODEL
model_roof = load_model('keras_model_roof.h5')
##### RENAME LAYERS TO BE UNIQUE
for layer in model_roof.layers:
    #if i==0:
    #    pass
    #else:
        layer._name = layer._name + str("_roof")
##### RENAME MODEL TO BE UNIQUE
model_roof._name = 'model_roof'
##### DROP INPUT LAYER
#model_roof._layers.pop(0)


##### CREATE TWO OUTPUT
output1 = model_roof(inputs)
output2 = model_cars(inputs)

##### CONCATENATE
outputs = concatenate([output1,output2], axis = 3)

##### BUILD MODEL AROUND COMMON INPUT AND OUTPUTS
merged = Model(inputs=[inputs],outputs=[outputs])

##### SAVE MERGED MODEL
merged.save('merged.h5')

"""
