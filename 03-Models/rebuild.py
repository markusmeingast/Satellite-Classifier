"""
This script builds a combined blank/untrained model of the two U-Net models.
The weights of the untrained model are then set by the two seperate trained
models.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

################################################################################
# %% UNET MODEL
################################################################################

def unet_model(power=2):

    ##### COMMON INPUT LAYER
    input_size = (256,256,3)
    inputs = Input(input_size)


    ##### UNET A
    conv1_a = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(inputs)
    conv1_a = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv1_a)
    pool1_a = MaxPooling2D(pool_size=(2, 2))(conv1_a)
    conv2_a = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool1_a)
    conv2_a = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv2_a)
    pool2_a = MaxPooling2D(pool_size=(2, 2))(conv2_a)
    conv3_a = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool2_a)
    conv3_a = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv3_a)
    pool3_a = MaxPooling2D(pool_size=(2, 2))(conv3_a)

    conv4_a = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool3_a)
    conv4_a = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv4_a)
    drop4_a = Dropout(0.5)(conv4_a)
    pool4_a = MaxPooling2D(pool_size=(2, 2))(drop4_a)

    conv5_a = Conv2D(2**(power+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool4_a)
    conv5_a = Conv2D(2**(power+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv5_a)
    drop5_a = Dropout(0.5)(conv5_a)

    up6_a = Conv2D(2**(power+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(drop5_a))
    merge6_a = concatenate([drop4_a,up6_a], axis = 3)
    conv6_a = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge6_a)
    conv6_a = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv6_a)

    up7_a = Conv2D(2**(power+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv6_a))
    merge7_a = concatenate([conv3_a,up7_a], axis = 3)
    conv7_a = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge7_a)
    conv7_a = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv7_a)

    up8_a = Conv2D(2**(power+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv7_a))
    merge8_a = concatenate([conv2_a,up8_a], axis = 3)
    conv8_a = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge8_a)
    conv8_a = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv8_a)

    up9_a = Conv2D(2**(power), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv8_a))
    merge9_a = concatenate([conv1_a,up9_a], axis = 3)
    conv9_a = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge9_a)
    conv9_a = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv9_a)
    conv9_a = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv9_a)
    conv10_a = Conv2D(1, 1, activation = 'sigmoid')(conv9_a)

    ##### UNET B
    conv1_b = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(inputs)
    conv1_b = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv1_b)
    pool1_b = MaxPooling2D(pool_size=(2, 2))(conv1_b)
    conv2_b = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool1_b)
    conv2_b = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv2_b)
    pool2_b = MaxPooling2D(pool_size=(2, 2))(conv2_b)
    conv3_b = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool2_b)
    conv3_b = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv3_b)
    pool3_b = MaxPooling2D(pool_size=(2, 2))(conv3_b)

    conv4_b = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool3_b)
    conv4_b = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv4_b)
    drop4_b = Dropout(0.5)(conv4_b)
    pool4_b = MaxPooling2D(pool_size=(2, 2))(drop4_b)

    conv5_b = Conv2D(2**(power+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool4_b)
    conv5_b = Conv2D(2**(power+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv5_b)
    drop5_b = Dropout(0.5)(conv5_b)

    up6_b = Conv2D(2**(power+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(drop5_b))
    merge6_b = concatenate([drop4_b,up6_b], axis = 3)
    conv6_b = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge6_b)
    conv6_b = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv6_b)

    up7_b = Conv2D(2**(power+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv6_b))
    merge7_b = concatenate([conv3_b,up7_b], axis = 3)
    conv7_b = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge7_b)
    conv7_b = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv7_b)

    up8_b = Conv2D(2**(power+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv7_b))
    merge8_b = concatenate([conv2_b,up8_b], axis = 3)
    conv8_b = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge8_b)
    conv8_b = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv8_b)

    up9_b = Conv2D(2**(power), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv8_b))
    merge9_b = concatenate([conv1_b,up9_b], axis = 3)
    conv9_b = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge9_b)
    conv9_b = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv9_b)
    conv9_b = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv9_b)
    conv10_b = Conv2D(1, 1, activation = 'sigmoid')(conv9_b)


    ##### UNET C
    conv1_c = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(inputs)
    conv1_c = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv1_c)
    pool1_c = MaxPooling2D(pool_size=(2, 2))(conv1_c)
    conv2_c = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool1_c)
    conv2_c = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv2_c)
    pool2_c = MaxPooling2D(pool_size=(2, 2))(conv2_c)
    conv3_c = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool2_c)
    conv3_c = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv3_c)
    pool3_c = MaxPooling2D(pool_size=(2, 2))(conv3_c)

    conv4_c = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool3_c)
    conv4_c = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv4_c)
    drop4_c = Dropout(0.5)(conv4_c)
    pool4_c = MaxPooling2D(pool_size=(2, 2))(drop4_c)

    conv5_c = Conv2D(2**(power+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(pool4_c)
    conv5_c = Conv2D(2**(power+4), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv5_c)
    drop5_c = Dropout(0.5)(conv5_c)

    up6_c = Conv2D(2**(power+3), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(drop5_c))
    merge6_c = concatenate([drop4_c,up6_c], axis = 3)
    conv6_c = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge6_c)
    conv6_c = Conv2D(2**(power+3), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv6_c)

    up7_c = Conv2D(2**(power+2), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv6_c))
    merge7_c = concatenate([conv3_c,up7_c], axis = 3)
    conv7_c = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge7_c)
    conv7_c = Conv2D(2**(power+2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv7_c)

    up8_c = Conv2D(2**(power+1), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv7_c))
    merge8_c = concatenate([conv2_c,up8_c], axis = 3)
    conv8_c = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge8_c)
    conv8_c = Conv2D(2**(power+1), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv8_c)

    up9_c = Conv2D(2**(power), 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(UpSampling2D(size = (2,2), interpolation='bilinear')(conv8_c))
    merge9_c = concatenate([conv1_c,up9_c], axis = 3)
    conv9_c = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(merge9_c)
    conv9_c = Conv2D(2**(power), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv9_c)
    conv9_c = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(conv9_c)
    conv10_c = Conv2D(1, 1, activation = 'sigmoid')(conv9_c)

    ##### CONCAT OUTPUTS FROM A AND B
    outputs = concatenate([conv10_a,conv10_b,conv10_c], axis = 3)

    ##### BUILD MODEL
    model = keras.Model(inputs = inputs, outputs = outputs)
    return model

##### BUILD MODEL
model = unet_model(4)

##### COMPILATION MAY NOT BE REQUIRED, AS NO FURTHER TRAINING IS DONE
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

##### SAVE MODEL FOR WEIGHTS TRANSFER
model.save('keras_model_merged_untrained.h5')
