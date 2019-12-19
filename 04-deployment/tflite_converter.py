"""
Converter script to produce tflite model that should be TPU compilable.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import tensorflow as tf

################################################################################
# %% NEED TO PROVIDE DATA AS EXAMPLE INPUT FOR CONVERSION/QUANTIFICATION
################################################################################

X_train = np.load('../00-raw-data/potsdam-dataset/pre/cars/image-train-0.npy')[0:100]

##### CONVERT DIMENSIONS FOR CONV2D
X_train = X_train/255


################################################################################
# %% CONVERT
################################################################################

##### GENERATOR FOR SAMPLE INPUT DATA TO QUANTIZE ON
def representative_dataset_gen():
    for i in range(100):
        yield [(X_train[i, None]).astype(np.float32)]

##### CREATE CONVERTER
#model = tf.keras.models.load_model('keras_model.h5')
#converter = tf.lite.TFLiteConverter.from_keras_model(model) # <-- ISSUES GETTING QUANTIZED!
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('../03-Models/keras_model_merged.h5')

##### SHOW MODEL WHAT DATA WILL LOOK LIKE
converter.representative_dataset = representative_dataset_gen

##### QUANTIZE INTERNALS TO UINT8
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

##### REDUCE ALL INTERNAL OPERATIONS TO UNIT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.inference_type = tf.float32

##### CONVERT THE MODEL
tflite_model = converter.convert()

##### SAVE MODEL TO FILE
tflite_model_name = "merged.tflite"
open(tflite_model_name, "wb").write(tflite_model)

##### MODEL SHOULD NOW BE COMPILED!
# : edgetpu_compiler -s mnist.tflite

################################################################################
# %% VARIOUS OPTIONS FOR EXPORTING...
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
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
