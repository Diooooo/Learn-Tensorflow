import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def vec2mat(vec):
    res = np.pad(vec.reshape(-1, 10, 10), ((0, 0), (9, 9), (9, 9)), 'constant', constant_values=0)
    np.random.shuffle(res)
    res = np.expand_dims(res, axis=3)
    return res


def generator(z, label, is_train, n_unit=32, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        z_concat = tf.concat([z, label], axis=1)
        h1 = tf.layers.dense(z_concat, 7 * 7 * 16)
        h1 = tf.layers.batch_normalization(h1)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.reshape(h1, shape=[-1, 7, 7, 16])
        h1 = tf.layers.conv2d_transpose(h1, n_unit, (3, 3), strides=[2, 2], padding='same')
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.layers.conv2d_transpose(h1, n_unit * 2, (3, 3), strides=[2, 2], padding='same')
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        logits = tf.layers.conv2d(h1, 1, (3, 3), padding='same', activation=tf.nn.sigmoid)
        return logits


def discriminator(x, label, is_train, n_unit, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        label_reshape = tf.layers.dense(label, 784)
        label_reshape = tf.reshape(label_reshape, shape=[-1, 28, 28, 1])
        x_concat = tf.concat([x, label_reshape], axis=3)
        h1 = tf.layers.conv2d(x_concat, n_unit, (3, 3), padding='same')
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.nn.dropout(h1, 0.5)
        h1 = tf.layers.conv2d(h1, n_unit / 2, (3, 3), padding='same')
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.nn.dropout(h1, 0.5)
        h1 = tf.layers.flatten(h1)
        logits = tf.layers.dense(h1, 1, activation=tf.nn.sigmoid)
        return logits


if __name__ == "__main__":
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    input_size = [28, 28]

    z_size = 100

    alpha = 0.2

    tf.reset_default_graph()

    z = tf.placeholder(tf.float32, shape=[None, z_size], name='z')
    target = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 1], name='target')
    label = tf.placeholder(tf.float32, shape=[None, 10], name='condition')
    is_train = tf.placeholder(tf.bool)

    g_logits = generator(z, label, is_train, n_unit=16, reuse=False, alpha=alpha)
    d_logits_real = discriminator(target, label, is_train, n_unit=32, reuse=False, alpha=alpha)
    d_logits_fake = discriminator(g_logits, label, is_train, n_unit=32, reuse=True, alpha=alpha)

    eps = 1e-2
    d_loss = -tf.reduce_mean(tf.log(d_logits_real + eps) + tf.log(1 - d_logits_fake + eps))

    # g_loss = -tf.reduce_mean(tf.log(d_logits_fake + eps))
    g_loss = tf.reduce_mean(tf.square(tf.log(d_logits_fake + eps)))

    learning_rate = 0.0001

    total_var = tf.trainable_variables()
    d_var = [v for v in total_var if v.name.startswith('discriminator')]
    g_var = [v for v in total_var if v.name.startswith('generator')]
    bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    bn_ops_d = [var for var in bn_ops if var.name.startswith('discriminator')]
    bn_ops_g = [var for var in bn_ops if var.name.startswith('generator')]

    with tf.control_dependencies(bn_ops_d):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=d_var)
    with tf.control_dependencies(bn_ops_g):
        g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate * 10).minimize(g_loss, var_list=g_var)

    batch_size = 128
    epochs = 50
    losses = []
    samples = []
    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('./logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for it in range(mnist.train.num_examples // batch_size):
                batch = mnist.train.next_batch(batch_size)

                batch_image = batch[0].reshape(batch_size, 28, 28, 1)
                # batch_image[batch_image > 0.5] = 1
                # batch_image[batch_image <= 0.5] = 0

                batch_label = batch[1]

                batch_z = np.random.uniform(low=-1, high=1, size=(batch_size, 100))

                loss_d, _ = sess.run([d_loss, d_train_opt],
                                     feed_dict={target: batch_image, z: batch_z, label: batch_label,
                                                is_train: True})  # G(z) as input of D
                loss_g, _ = sess.run([g_loss, g_train_opt],
                                     feed_dict={z: batch_z, label: batch_label, is_train: True})  # generate G(z)
                print('iteration: {}/{}, d_loss: {}, g_loss: {}'.format(it, mnist.train.num_examples // batch_size,
                                                                        loss_d, loss_g))
                losses.append([loss_d, loss_g])
            print('*' * 10,
                  'Epoch {}/{} ...'.format(epoch + 1, epochs),
                  'd_loss: {:.4f} ...'.format(loss_d),
                  'g_loss: {:.4f} ...'.format(loss_g),
                  '*' * 10)

            sample_z = np.random.uniform(-1, 1, size=(10, 100))
            sample_label = np.zeros((10, 10))
            for i in range(10):
                sample_label[i, i] = 1
            gen_samples = sess.run(g_logits,
                                   feed_dict={z: sample_z, label: sample_label, is_train: False})
            samples.append(gen_samples)
    np.save('./losses-bsgan-c.npy', np.asarray(losses))
    np.save('./samples-bsgan-c.npy', np.asarray(samples))

    losses = np.asarray(losses)
    samples = np.asarray(samples)
    plt.plot(losses[:, 0])
    plt.title('d loss')
    plt.show()
    plt.plot(losses[:, 1])
    plt.title('g loss')
    plt.show()

    # for index in range(2):
    # samples = np.load('./samples-gan-disc.npy')
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(samples[-1, i].reshape(28, 28), cmap='gray')
    plt.show()
