import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def fc_layer(input, name, shape):
    with tf.name_scope(name) as scope:
        W = tf.get_variable(name=name + '_W', shape=[shape[0], shape[1]],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        b = tf.get_variable(name=name + '_b', shape=[shape[1]], initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        x_norm = tf.layers.batch_normalization(tf.matmul(input, W) + b, training=True, name=name + '_BN')
        return tf.nn.relu(x_norm)


def loss_function(logits, labels, name):
    with tf.name_scope(name) as scope:
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=name + '_loss'))

    return loss


def optimizer(learning_rate, loss, name):
    with tf.name_scope(name) as scope:
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)
    return opt


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    learning_rate = 0.01
    batch_size = 128
    epochs = 50

    X = tf.placeholder(tf.float32, [batch_size, 784])
    Y = tf.placeholder(tf.float32, [batch_size, 10])

    layer1 = fc_layer(X, 'input', (784, 1024))
    layer2 = fc_layer(layer1, 'hidden1', (1024, 1024))
    layer3 = fc_layer(layer2, 'hidden2', (1024, 1024))
    logits = fc_layer(layer3, 'output', (1024, 10))

    loss = loss_function(logits, Y, 'softmax')
    opt = optimizer(learning_rate, loss, 'sgd')
