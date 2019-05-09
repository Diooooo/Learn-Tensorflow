import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def model_inputs(real_dim, z_dim):
    input_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    return input_real, input_z


def generator(z, output_dim, is_train, n_unit=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_unit)
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.layers.dense(h1, n_unit * 2)
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.layers.dense(h1, n_unit * 4)
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.layers.dense(h1, n_unit * 8)
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        logits = tf.layers.dense(h1, output_dim)
        out = tf.nn.sigmoid(logits)
        return logits, out


def discriminator(x, is_train, n_unit=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_unit)
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.layers.dense(h1, n_unit / 2)
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.layers.dense(x, n_unit / 4)
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        logits = tf.layers.dense(h1, 1)
        out = tf.nn.sigmoid(logits)
        return logits, out


if __name__ == "__main__":
    mnist = input_data.read_data_sets('./MNIST_data')

    input_size = 784

    z_size = 100

    g_hidden = 128
    d_hidden = 128

    alpha = 0.01
    smooth = 0.1  # reduce label to (1-smooth), help training

    tf.reset_default_graph()

    input_real, input_z = model_inputs(input_size, z_size)
    is_train = tf.placeholder(tf.bool)
    g_logits, g_model = generator(input_z, input_size, is_train, n_unit=g_hidden, reuse=False, alpha=alpha)
    d_logits_real, d_model_real = discriminator(input_real, is_train, n_unit=d_hidden, reuse=False, alpha=alpha)
    d_logits_fake, d_model_fake = discriminator(g_model, is_train, n_unit=d_hidden, reuse=True, alpha=alpha)

    eps = 1e-16
    d_loss = -tf.reduce_mean(tf.log(d_model_real + eps) + tf.log(1 - d_model_fake + eps))

    g_loss = -tf.reduce_mean(tf.log(d_model_fake + eps))

    learning_rate = 0.01

    total_var = tf.trainable_variables()
    d_var = [v for v in total_var if v.name.startswith('discriminator')]
    g_var = [v for v in total_var if v.name.startswith('generator')]
    bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    bn_ops_d = [var for var in bn_ops if var.name.startswith('discriminator')]
    bn_ops_g = [var for var in bn_ops if var.name.startswith('generator')]

    with tf.control_dependencies(bn_ops_d):
        d_train_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_var)
    with tf.control_dependencies(bn_ops_g):
        g_train_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_var)

    batch_size = 100
    epochs = 10
    losses = []
    samples = []
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for _ in range(mnist.train.num_examples // batch_size):
                batch = mnist.train.next_batch(batch_size)

                batch_image = batch[0].reshape((batch_size, 784))
                batch_image = batch_image * 2 - 1  # (0~1) to (-1, 1)

                batch_z = np.random.uniform(low=-1, high=1, size=(batch_size, z_size))

                loss_d, _ = sess.run([d_loss, d_train_opt],
                                     feed_dict={input_real: batch_image, input_z: batch_z,
                                                is_train: True})  # G(z) as input of D
                loss_g, _ = sess.run([g_loss, g_train_opt],
                                     feed_dict={input_z: batch_z, is_train: True})  # generate G(z)
                print('d_loss: {}, g_loss: {}'.format(loss_d, loss_g))
                losses.append([loss_d, loss_g])
            print('Epoch {}/{} ...'.format(epoch + 1, epochs),
                  'd_loss: {:.4f} ...'.format(loss_d),
                  'g_loss: {:.4f} ...'.format(loss_g))

            sample_z = np.random.uniform(-1, 1, size=(16, z_size))
            gen_samples = sess.run(g_model,
                                   feed_dict={input_z: sample_z, is_train: False})
            samples.append(gen_samples)
    np.save('losses.npy', np.asarray(losses))
    np.save('samples.npy', np.asarray(samples))

    losses = np.asarray(losses)
    samples = np.asarray(samples)
    plt.plot(losses[:, 0])
    plt.title('d loss')
    plt.show()
    plt.plot(losses[:, 1])
    plt.title('g loss')
    plt.show()

    plt.figure()
    index = 5
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(samples[i, index].reshape(28, 28), cmap='gray')
    plt.show()