"""



CLASS LABELS
1	E6004D	Artificial surfaces > Urban fabric > Continuous urban fabric
2	FF0000	Artificial surfaces > Urban fabric > Discontinuous urban fabric
3	CC4DF2	Artificial surfaces > Industrial, commercial, and transport units > Industrial or commercial units
4	CC0000	Artificial surfaces > Industrial, commercial, and transport units > Road and rail networks and associated land
5	E6CCCC	Artificial surfaces > Industrial, commercial, and transport units > Port areas
6	E6CCE6	Artificial surfaces > Industrial, commercial, and transport units > Airports
7	A600CC	Artificial surfaces > Mine, dump, and construction sites > Mineral extraction sites
8	A64DCC	Artificial surfaces > Mine, dump, and construction sites > Dump sites
9	FF4DFF	Artificial surfaces > Mine, dump, and construction sites > Construction sites
10	FFA6FF	Artificial surfaces > Artificial, non-agricultural vegetated areas > Green urban areas
11	FFE6FF	Artificial surfaces > Artificial, non-agricultural vegetated areas > Sport and leisure facilities
12	FFFFA8	Agricultural areas > Arable land > Non-irrigated arable land
13	FFFF00	Agricultural areas > Arable land > Permanently irrigated land
14	E6E600	Agricultural areas > Arable land > Rice fields
15	E68000	Agricultural areas > Permanent crops > Vineyards
16	F2A64D	Agricultural areas > Permanent crops > Fruit trees and berry plantations
17	E6A600	Agricultural areas > Permanent crops > Olive groves
18	E6E64D	Agricultural areas > Pastures > Pastures
19	FFE6A6	Agricultural areas > Heterogeneous agricultural areas > Annual crops associated with permanent crops
20	FFE64D	Agricultural areas > Heterogeneous agricultural areas > Complex cultivation patterns
21	E6CC4D	Agricultural areas > Heterogeneous agricultural areas > Land principally occupied by agriculture, with significant areas of natural vegetation
22	F2CCA6	Agricultural areas > Heterogeneous agricultural areas > Agro-forestry areas
23	80FF00	Forest and semi natural areas > Forests > Broad-leaved forest
24	00A600	Forest and semi natural areas > Forests > Coniferous forest
25	4DFF00	Forest and semi natural areas > Forests > Mixed forest
26	CCF24D	Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Natural grasslands
27	A6FF80	Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Moors and heathland
28	A6E64D	Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Sclerophyllous vegetation
29	A6F200	Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Transitional woodland-shrub
30	E6E6E6	Forest and semi natural areas > Open spaces with little or no vegetation > Beaches, dunes, sands
31	CCCCCC	Forest and semi natural areas > Open spaces with little or no vegetation > Bare rocks
32	CCFFCC	Forest and semi natural areas > Open spaces with little or no vegetation > Sparsely vegetated areas
33	000000	Forest and semi natural areas > Open spaces with little or no vegetation > Burnt areas
34	A6E6CC	Forest and semi natural areas > Open spaces with little or no vegetation > Glaciers and perpetual snow
35	A6A6FF	Wetlands > Inland wetlands > Inland marshes
36	4D4DFF	Wetlands > Inland wetlands > Peat bogs
37	CCCCFF	Wetlands > Maritime wetlands > Salt marshes
38	E6E6FF	Wetlands > Maritime wetlands > Salines
39	A6A6E6	Wetlands > Maritime wetlands > Intertidal flats
40	00CCF2	Water bodies > Inland waters > Water courses
41	80F2E6	Water bodies > Inland waters > Water bodies
42	00FFA6	Water bodies > Marine waters > Coastal lagoons
43	A6FFE6	Water bodies > Marine waters > Estuaries
44	E6F2FF	Water bodies > Marine waters > Sea and ocean
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from time import time
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import cv2
import pandas as pd

################################################################################
# %% DEFINE SOLVER PARAMETERS
################################################################################

batch_size = 1000
training_ratio = 0.8

################################################################################
# %% IMPORT IMAGE AND TARGET DATA
################################################################################

#input_a = np.load('../02-preprocess/input-berlin.npy')
input_b = np.load('../02-preprocess/input-bordeaux.npy')
input_c = np.load('../02-preprocess/input-madrid.npy')
#input_d = np.load('../02-preprocess/input-oslo.npy')
#input_e = np.load('../02-preprocess/input-lisbon.npy')

target_a = np.load('../02-preprocess/target-berlin.npy')
target_b = np.load('../02-preprocess/target-bordeaux.npy')
target_c = np.load('../02-preprocess/target-madrid.npy')
target_d = np.load('../02-preprocess/target-oslo.npy')
target_e = np.load('../02-preprocess/target-lisbon.npy')

#data_input = np.concatenate((input_a, input_b, input_c, input_d, input_e))
data_target = np.concatenate((target_a, target_b, target_c, target_d, target_e))

data_input = np.concatenate((input_b, input_c))
data_target = np.concatenate((target_b, target_c))


len(data_target)

################################################################################
# %% INVESTIGATE CLASS IMBALANCE
################################################################################

df = pd.DataFrame(data_target, columns=['cat'])
df['dummy'] = np.ones((len(data_target),1))
ordered = (df.groupby('cat').sum()).sort_values('dummy', ascending=False)

ordered

regroup = {
    1:'A', 2:'A',
    3:'B', 4:'B', 5:'B', 6:'B',
    7:'C', 8:'C', 9:'C',
    10:'D', 11:'D',
    12:'E', 13:'E', 14:'E',
    15:'F', 16:'F', 17:'F',
    18:'G',
    19:'H', 20:'H', 21:'H', 22:'H',
    23:'I', 24:'I', 25:'I',
    26:'J', 27:'J', 28:'J', 29:'J',
    30:'X', 31:'X', 32:'X', 33:'X',
    34:'X', 35:'X', 36:'X', 37:'X', 38:'X', 39:'X',
    40:'K', 41:'K', 42:'K', 43:'K', 44:'K'
}


ordered.reindex(np.arange(1,45))

##### PUT ALL CLASSES WITH <5% REPRESENTATION INTO 45 (OTHER)
cat_drop = np.array([26, 28, 10, 1, 40, 22, 6, 4, 27, 9, 7, 17, 14, 16, 37, 36, 32, 35, 8, 38, 39, 5, 30, 42, 19, 33])
for idx, cat in enumerate(data_target):
    if cat in cat_drop:
        data_target[idx] = 45

################################################################################
# %% TRAINING AND TESTING SPLIT
################################################################################

ind = np.arange(len(data_input))
ind = np.random.shuffle(ind)

X_train = data_input[:int(training_ratio*len(data_input)),:,:,:]
X_test = data_input[int(training_ratio*len(data_input)):,:,:,:]

y_train = data_target[:int(training_ratio*len(data_input))]
y_test = data_target[int(training_ratio*len(data_input)):]

##### CONVERT DIMENSIONS FOR CONV2D
X_train = X_train/255
X_test = X_test/255

##### ONE HOT ENCODE
##### ON TRAINING DATA
y_train_ohe = np.zeros(((y_train.size, 45)))
y_train_ohe[np.arange(y_train.size),y_train-1] = 1

##### ON TESTING DATA
y_test_ohe = np.zeros(((y_test.size, 45)))
y_test_ohe[np.arange(y_test.size),y_test-1] = 1

##### DROP EMPTY COLUMNS
y_train_ohe = np.delete(y_train_ohe,cat_drop-1,axis=1)
y_test_ohe = np.delete(y_test_ohe,cat_drop-1,axis=1)



################################################################################
# %% BUILD MODEL
################################################################################

##### GET INPUT SHAPE
(_, xdim, ydim, zdim) = X_train.shape

##### ADD LAYERS
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(xdim,ydim,zdim), dtype=tf.float32))
model.add(keras.layers.Conv2D(4, (3, 3)))
model.add(keras.layers.Conv2D(10, (3, 3), activation='relu'))
#model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(10, activation='relu'))
#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(44-len(cat_drop)+1, activation='softmax'))

##### COMPILE MODEL AND PRINT SUMMARY
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

##### PRINT SUMMARY
print(model.summary())

################################################################################
# %% INIT CALLBACKS
################################################################################

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), update_freq='epoch')
earlystopping = EarlyStopping(monitor='val_loss', patience=3)

################################################################################
# %% RUN MODEL
#%%time
################################################################################

history = model.fit(
    x=X_train,
    y=y_train_ohe,
    epochs=100,
    verbose=1,
    validation_data=(X_test, y_test_ohe),
    use_multiprocessing=True,
    batch_size=batch_size,
    callbacks=[tensorboard]#, earlystopping]
)

model.save('keras_model.h5')

##### POSSIBLE RESTART POINT
#model = keras.models.load_model('keras_model.h5')

################################################################################
# %% MODEL PERFORMANCES
################################################################################

#
# C2D(1)                          -> FLT               -> D(44)   22% / 17% (ES)
# C2D(4)                          -> FLT               -> D(44)   57% / 67% (ES)
# C2D(10)                         -> FLT               -> D(44)   58% / 68% (ES)
# C2D(10) -> C2D(10,relu)         -> FLT               -> D(44)   66% / 73% (ES)
# C2D(10) -> C2D(10,relu) -> MaxP -> FLT               -> D(44)   61% / 71% (ES)
# C2D(10) -> C2D(10,relu) -> DO25 -> FLT -> D(10,relu) -> D(44)   63% / 72% (KL)

################################################################################
# %% PLOT LOSS HISTORY
################################################################################

mp.semilogy(history.history['loss'], label='Training')
mp.semilogy(history.history['val_loss'], label='Testing')
mp.legend()
mp.show()

################################################################################
# %% CONVERT
################################################################################

def representative_dataset_gen():
    for i in range(100):
        yield [X_train[i, None].astype(np.float32)]

##### CREATE CONVERTER
#converter = tf.lite.TFLiteConverter.from_keras_model(model) # <-- ISSUES GETTING QUANTIZED!
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('keras_model.h5')

##### SHOW MODEL WHAT DATA WILL LOOK LIKE
converter.representative_dataset = representative_dataset_gen

##### QUANTIZE INTERNALS TO UINT8
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

##### REDUCE ALL INTERNAL OPERATIONS TO UNIT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
converter.inference_type = tf.float32

##### CONVERT THE MODEL
tflite_model = converter.convert()

##### SAVE MODEL TO FILE
tflite_model_name = "mnist.tflite"
open(tflite_model_name, "wb").write(tflite_model)

################################################################################
# %% OPTIONS DUMP
################################################################################

#converter.representative_dataset = representative_dataset_gen
#converter.input_shapes= (1,28,28,1)
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
#converter.inference_type = tf.float32
#converter.std_dev_values = 0.3
#converter.mean_values = 0.5
#converter.default_ranges_min = 0.0
#converter.default_ranges_max = 1.0
#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                       tf.lite.OpsSet.SELECT_TF_OPS]
#converter.post_training_quantize=True
#    --input_arrays=conv2d_input \
#    --output_arrays=dense/Softmax \

#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

##### SET DEFAULT OPTIMIZATIONS

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
