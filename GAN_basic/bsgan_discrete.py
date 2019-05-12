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


def log_sum_exp(x, axis=None):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    y = tf.exp(x - x_max)
    y = tf.reduce_sum(y, axis=axis, keepdims=True)
    return tf.log(y) + x_max


def generator(z, label, is_train, n_unit=32, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        z_concat = tf.concat([z, label], axis=1)
        h1 = tf.layers.dense(z_concat, 1024)
        h1 = tf.layers.batch_normalization(h1)
        # h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.layers.dense(h1, 7 * 7 * n_unit * 2)
        h1 = tf.layers.batch_normalization(h1)
        # h1 = tf.nn.leaky_relu(h1, alpha)
        h1 = tf.reshape(h1, shape=[-1, 7, 7, n_unit * 2])
        h1 = tf.layers.conv2d_transpose(h1, n_unit, (5, 5), strides=[2, 2], padding='same')
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        # h1 = tf.nn.leaky_relu(h1, alpha)
        # h1 = tf.layers.conv2d_transpose(h1, n_unit * 2, (3, 3), strides=[2, 2], padding='same')
        # h1 = tf.layers.batch_normalization(h1, training=is_train)
        # h1 = tf.nn.leaky_relu(h1, alpha)
        # h1 = tf.layers.conv2d(h1, n_unit * 4, (3, 3), padding='same')
        # h1 = tf.layers.batch_normalization(h1, training=is_train)
        # h1 = tf.nn.leaky_relu(h1, alpha)
        # h1 = tf.layers.conv2d(h1, n_unit * 8, (3, 3), padding='same')
        # h1 = tf.layers.batch_normalization(h1, training=is_train)
        # h1 = tf.nn.leaky_relu(h1, alpha)
        # logits = tf.layers.conv2d(h1, 1, (3, 3), padding='same')
        logits = tf.layers.conv2d_transpose(h1, 1, (5, 5), strides=[2, 2], padding='same')
        return logits, tf.nn.sigmoid(logits)


def discriminator(x, label, is_train, n_unit, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        label_reshape = tf.layers.dense(label, 784)
        label_reshape = tf.reshape(label_reshape, shape=[-1, 28, 28, 1])
        x_concat = tf.concat([x, label_reshape], axis=3)
        h1 = tf.layers.conv2d(x_concat, n_unit, (5, 5), padding='same')
        # h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        # h1 = tf.nn.dropout(h1, 0.5)
        # h1 = tf.layers.max_pooling2d(h1, (2, 2), 2)
        h1 = tf.layers.conv2d(h1, n_unit * 2, (5, 5), padding='same')
        # h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.nn.leaky_relu(h1, alpha)
        # h1 = tf.nn.dropout(h1, 0.5)
        # h1 = tf.layers.max_pooling2d(h1, (2, 2), 2)
        h1 = tf.layers.flatten(h1)
        h1 = tf.layers.dense(h1, 1024)
        h1 = tf.nn.leaky_relu(h1, alpha)
        logits = tf.layers.dense(h1, 1)
        return logits


if __name__ == "__main__":
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    input_size = [28, 28]

    z_size = 100

    M = 20

    alpha = 0.2

    batch_size = 32

    eps = 1e-7

    g_units = 64
    d_units = 64

    tf.reset_default_graph()

    z = tf.placeholder(tf.float32, shape=[batch_size, z_size], name='z')
    target = tf.placeholder(tf.float32, shape=[batch_size, input_size[0], input_size[1], 1], name='target')
    label = tf.placeholder(tf.float32, shape=[batch_size, 10], name='condition')
    is_train = tf.placeholder(tf.bool)

    _, g_prob = generator(z, label, is_train, n_unit=g_units, reuse=False, alpha=alpha)

    d_logits_real = discriminator(target, label, is_train, n_unit=d_units, reuse=False, alpha=alpha)

    rand = tf.random_uniform(shape=[M, batch_size, input_size[0], input_size[1], 1], minval=0, maxval=1)
    g_samples = tf.cast(rand <= tf.expand_dims(g_prob, axis=0), tf.float32)
    g_samples_reshape = tf.reshape(g_samples, shape=[-1, input_size[0], input_size[1], 1])
    label_samples = tf.tile(tf.expand_dims(label, axis=0), multiples=[M, 1, 1])
    label_samples_reshape = tf.reshape(label_samples, shape=[-1, 10])
    d_logits_fake = discriminator(g_samples_reshape, label_samples_reshape, is_train,
                                  n_unit=d_units, reuse=True, alpha=alpha)

    d_logits_fake = tf.reshape(d_logits_fake, shape=[M, -1])

    # weights = tf.exp(d_logits_fake)
    # weights_norm = (weights + eps) / (tf.reduce_sum(weights, axis=0) + eps)
    # weights_norm = tf.stop_gradient(weights_norm)

    log_N = tf.log(tf.cast(d_logits_fake.shape[0], dtype=tf.float32))
    log_Z_est = log_sum_exp(d_logits_fake - log_N, axis=0)
    log_w_tilde = d_logits_fake - log_Z_est - log_N
    w_tilde = tf.exp(log_w_tilde)
    w_tilde = tf.stop_gradient(w_tilde)

    # d_loss = -tf.reduce_mean(tf.log(d_logits_real + eps) + tf.log(1 - d_logits_fake + eps))
    # d_loss = tf.reduce_mean(weights_norm) - tf.reduce_mean(d_logits_real)
    d_loss = tf.reduce_mean(tf.nn.softplus(-d_logits_fake)) + tf.reduce_mean(d_logits_fake) + tf.reduce_mean(
        tf.nn.softplus(-d_logits_real))

    # g_loss = -tf.reduce_mean(tf.log(d_logits_fake + eps))
    # g_loss = tf.reduce_mean(tf.square(tf.log(d_logits_fake + eps)))
    # log_g = tf.log(g_logits + eps)
    g_logits_expand = tf.expand_dims(tf.clip_by_value(g_prob, eps, 1 - eps), axis=0)
    log_g = tf.reduce_sum(g_samples * tf.log(g_logits_expand) + (1 - g_samples) * tf.log(1 - g_logits_expand),
                          axis=[2, 3, 4])
    # g_loss = -tf.reduce_mean(tf.reduce_sum(weights_norm * tf.expand_dims(log_g, axis=0), axis=[1, 2, 3, 4]))
    g_loss = -tf.reduce_mean(tf.reduce_sum(w_tilde * log_g, axis=0))

    learning_rate = 0.0001

    total_var = tf.trainable_variables()
    d_var = [v for v in total_var if v.name.startswith('discriminator')]
    g_var = [v for v in total_var if v.name.startswith('generator')]
    bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    bn_ops_d = [var for var in bn_ops if var.name.startswith('discriminator')]
    bn_ops_g = [var for var in bn_ops if var.name.startswith('generator')]

    with tf.control_dependencies(bn_ops_d):
        d_train_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate * 2).minimize(d_loss, var_list=d_var)
    with tf.control_dependencies(bn_ops_g):
        g_train_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_var)

    epochs = 5
    losses = []
    samples = []
    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('./logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for it in range(mnist.train.num_examples // batch_size):
                batch = mnist.train.next_batch(batch_size)

                batch_image = batch[0].reshape(batch_size, 28, 28, 1)
                batch_image[batch_image > 0.5] = 1
                batch_image[batch_image <= 0.5] = 0

                batch_label = batch[1]

                # batch_z = np.random.uniform(low=-1, high=1, size=(batch_size, 100))
                batch_z = np.random.normal(0, 1, size=(batch_size, 100)).astype(np.float32)
                # batch_image_z = vec2mat(batch_z)

                for _ in range(2):
                    loss_d, _ = sess.run([d_loss, d_train_opt],
                                         feed_dict={target: batch_image, z: batch_z, label: batch_label,
                                                    is_train: True})  # G(z) as input of D
                for _ in range(1):
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

            # sample_z = np.random.uniform(-1, 1, size=(128, 100))
            sample_z = np.random.normal(0, 1, size=(batch_size, 100)).astype(np.float32)
            sample_label = np.zeros((batch_size, 10))
            rand_index = np.random.randint(low=0, high=10, size=batch_size)
            for i in range(10):
                sample_label[i, rand_index[i]] = 1
            generated = sess.run(g_prob,
                                 feed_dict={z: sample_z, label: sample_label, is_train: False})
            samples.append(generated)
    np.save('./losses-bsgan-disc-cccc.npy', np.asarray(losses))
    np.save('./samples-bsgan-disc-cccc.npy', np.asarray(samples))

    losses = np.asarray(losses)
    samples = np.asarray(samples)
    plt.plot(losses[:, 0])
    plt.title('d loss')
    plt.show()
    plt.plot(losses[:, 1])
    plt.title('g loss')
    plt.show()

    # for index in range(2):
    # samples = np.load('./samples-bsgan-disc-cc.npy')
    plt.figure()
    # plt.axis('off')
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        img = samples[i, 0].reshape(28, 28)
        img[img > 0.5] = 1
        img[img < 0.5] = 0
        plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()
