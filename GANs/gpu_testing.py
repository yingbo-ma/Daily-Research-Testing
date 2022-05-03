# this file is for testing the existence of any local GPU devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
tf.test.is_gpu_available()

physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')
tf.config.experimental.set_memory_growth(physical_device[0],True)