import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class UNet:
    def __init__(self):
        pass

    def cnn_encode_layer(self, input, channels, scope_name, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            layer = tf.layers.conv2d(inputs=input, filters=channels, kernel_size=(3, 3), padding='same')
