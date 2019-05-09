import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def encoder(input):
    with tf.variable_scope('encoder'):
        conv1 = tf.layers.conv2d(inpyts=input, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                 name='conv_1')
        maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='same',
                                           name='maxpool_1')
        conv2 = tf.layers.conv2d(inpyts=maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                 name='conv_2')
        maxpool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2), padding='same',
                                           name='maxpool_2')
        conv3 = tf.layers.conv2d(inpyts=input, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                 name='conv_3')
        encode = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2), padding='same',
                                         name='maxpool_3')
        return encode


def decoder(encoder):
    with tf.variable_scope('decoder'):
        upsample1 = tf.image.resize_images(images=encoder, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv4 = tf.layers.conv2d(upsample1, 8, (3, 3), padding='same', activation=tf.nn.relu, name='deconv_1')
        upsample2 = tf.image.resize_images(images=conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.layers.conv2d(upsample2, 8, (3, 3), padding='same', activation=tf.nn.relu, name='deconv_2')
        upsample3 = tf.image.resize_images(images=conv5, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv6 = tf.layers.conv2d(upsample3, 8, (3, 3), padding='same', activation=tf.nn.relu, name='deconv_3')
        logits = tf.layers.conv2d(conv6, 1, (3, 3), padding='same')
        decoded = tf.nn.sigmoid(logits, name='decoded')
        return logits, decoded


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('./MNIST_data')
    inputs = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
    targets = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

    encode = encoder(inputs)
    logits, decoded = decoder(encode)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

