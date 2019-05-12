import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generator(X, variables, reuse=False):
    with tf.variable_scope('generator'):
        l = dense_layer(X, 100, 256, 'g1', variables)
        l = dense_layer(l, 256, 512, 'g2', variables)
        g = dense_layer(l, 512, 784, 'g3', variables)
    return g, tf.nn.sigmoid(g)


def discriminator(X, variables, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        l = dense_layer(X, 784, 512, 'd1', variables)
        l = dense_layer(l, 512, 256, 'd2', variables)
        d = dense_layer(l, 256, 1, 'd3', variables, activation=None)
    return d


def dense_layer(input, input_num, out_num, name, variables, activation='relu', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', shape=[input_num, out_num])
        b = tf.get_variable('b', shape=[out_num])
        if reuse is False:
            variables.append(w)
            variables.append(b)
        l = tf.matmul(input, w) + b
    if not activation:
        return l
    elif activation == 'relu':
        l = tf.nn.leaky_relu(l)
    elif activation == 'sigmoid':
        l = tf.nn.sigmoid(l)
    return l


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('./MNIST_data')

    M = 10
    X = tf.placeholder(tf.float32, shape=[None, 100], name='z')  # N samples
    X_sample = tf.placeholder(tf.float32, shape=[None, 784], name='z_sample')  # M*N samples
    Y = tf.placeholder(tf.float32, shape=[None, 784], name='target')  # N samples

    rd = 0.001
    rg = 0.001

    eps = 1e-16

    variables = []
    g_logits, g_prob = generator(X, variables)
    d_logits_real = discriminator(Y, variables)  # N

    d_logits_fake = discriminator(X_sample, variables, reuse=True)  # M*N

    weights = tf.exp(d_logits_fake)  # M*N
    weights = tf.reshape(weights, shape=[M, -1])
    weights_norm = (weights + eps) / (tf.reduce_sum(weights, axis=0) + eps)  # M*N

    total_var = tf.trainable_variables()
    d_var = [v for v in total_var if v.name.startswith('discriminator')]
    g_var = [v for v in total_var if v.name.startswith('generator')]

    vlb = tf.reduce_mean(d_logits_real) - tf.reduce_mean(weights)

    vlb_grad = tf.gradients(vlb, d_var, name='gradients_vlb')

    # update_d = [d_var[i].assign(d_var[i] + rd * vlb_grad[i]) for i in range(len(d_var))]
    update_d = tf.train.AdamOptimizer().minimize(-vlb, var_list=d_var)

    # log_g = tf.reduce_mean(tf.log(g_logits + eps))
    sample_g = tf.reshape(X_sample, shape=[M, -1, 784])
    g_logits_expand = tf.expand_dims(g_logits, axis=0)
    log_g = -tf.reduce_sum((1 - sample_g) * g_logits_expand + tf.nn.softplus(-g_logits_expand), axis=[2, 3, 4])

    log_g_grad = tf.gradients(log_g, g_var)

    w_g = tf.reduce_mean(tf.reduce_sum(weights_norm, axis=0))

    g_loss = tf.reduce_mean(tf.reduce_sum(w_g * log_g, axis=0))
    # update_g = [g_var[i].assign(g_var[i] + rg * (-w_g * log_g_grad[i])) for i in range(len(g_var))]
    update_g = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_var)
    epochs = 200
    m = 10
    batch_size = 128

    res = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            batch = mnist.train.next_batch(batch_size)
            batch_image = batch[0].reshape((batch_size, 784))
            batch_image[batch_image >= 0.5] = 1
            batch_image[batch_image < 0.5] = 0
            batch_z = np.random.uniform(low=-1, high=1, size=(batch_size, 100))
            batch_z_sample = np.random.uniform(low=-1, high=1, size=(m, batch_size, 784))

            prob_g = sess.run(g_prob, feed_dict={X: batch_z})
            batch_sample = np.zeros((m, batch_size, 784))
            for i in range(batch_size):
                for j in range(m):
                    batch_sample[j, i] = 1 * (batch_z_sample[j, i] <= prob_g[i])
            batch_sample = batch_sample.reshape(-1, 784)
            variational_lower_bound, _ = sess.run([vlb, update_d],
                                                  feed_dict={X: batch_z, Y: batch_image, X_sample: batch_sample})
            loss_g, _ = sess.run([g_loss, update_g],
                                 feed_dict={X: batch_z, Y: batch_image, X_sample: batch_sample})

            print('epoch: {}/{}, vlb: {}, g_loss: {}'.format(epoch + 1, epochs, variational_lower_bound, loss_g))
            if (epoch + 1) % 10 == 0:
                predicted = sess.run(g_prob, feed_dict={X: batch_z})
                res.append(predicted)
    res = np.asarray(res)
    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(res[-1, i].reshape(28, 28), cmap='gray')
    plt.show()
