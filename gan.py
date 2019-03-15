import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class GAN:
    def __init__(self):
        self.z = tf.placeholder(tf.float32, shape=[None, 2], name='z')  # input (of generator)
        self.x = tf.placeholder(tf.float32, shape=[None, 2], name='real_x')  # ground truth

        self.fake_x = self.netG(self.z)  # generated data
        self.real_logits = self.netD(self.x, reuse=False)  # use for the first time
        self.fake_logits = self.netD(self.fake_x,
                                     reuse=True)  # use the same parameters to compute loss for both real and fake

        # reduce_mean: compute mean on specific axis(if given)
        # sigmoid_...: z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # x:logits(output of the last layer, before input it into softmax layer), z:labels
        # sigmoid(x): predicted label
        # discriminator: try to maximize distance between real and fake(if so, loss_D is small)
        self.loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,
                                                                             labels=tf.ones_like(self.real_logits))) + \
                      tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                             labels=tf.zeros_like(self.real_logits)))
        # generator: try to minimize the distance between fake and real
        # (label of fake is 0, so goal is minimize fake and 1)
        self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                             labels=tf.ones_like(self.real_logits)))

        # we should know which parameter to be optimized
        t_var = tf.trainable_variables()
        self.d_var = [var for var in t_var if 'd_' in var.name]
        self.g_var = [var for var in t_var if 'g_' in var.name]

    def netG(self, z):
        with tf.variable_scope("generator") as scope:  # used for variables sharing
            W = tf.get_variable(name='g_W', shape=[2, 2], initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=True)
            b = tf.get_variable(name='g_b', shape=[2], initializer=tf.zeros_initializer(), trainable=True)
            W2 = tf.get_variable(name='g_W2', shape=[2, 2], initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b2 = tf.get_variable(name='g_b2', shape=[2], initializer=tf.zeros_initializer(), trainable=True)

            layer1 = tf.nn.tanh(tf.matmul(z, W) + b)
            return tf.matmul(layer1, W2) + b2

    def netD(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            W1 = tf.get_variable(name='d_W1', shape=[2, 5], initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b1 = tf.get_variable(name='d_b1', shape=[5], initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            W2 = tf.get_variable(name='d_W2', shape=[5, 3], initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b2 = tf.get_variable(name='d_b2', shape=[3], initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            W3 = tf.get_variable(name='d_W3', shape=[3, 1], initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b3 = tf.get_variable(name='d_b3', shape=[1], initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)

            layer1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
            return tf.matmul(layer2, W3) + b3


def iterate_minibatch(x, batch_size, shuffle=True):
    indices = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, x.shape[0], batch_size):
        yield x[indices[i:i + batch_size], :]


if __name__ == "__main__":
    # Generate "real" data
    X = np.random.normal(size=(1000, 2))
    A = np.array([[1, 2], [-0.1, 0.5]])
    b = np.array([1, 2])
    X = np.dot(X, A) + b

    # and stick them into an iterator
    batch_size = 4

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    gan = GAN()
    d_optim = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss=gan.loss_D, var_list=gan.d_var)
    g_optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=gan.loss_G, var_list=gan.g_var)

    init = tf.global_variables_initializer()  # is a must

    # config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(10):
            avg_loss = 0
            count = 0
            for x_batch in iterate_minibatch(X, batch_size):
                # generate random input(noise) for generator
                z_batch = np.random.normal(size=(4, 2))

                loss_D, _ = sess.run([gan.loss_D, d_optim], feed_dict={gan.z: z_batch, gan.x: x_batch})
                loss_G, _ = sess.run([gan.loss_G, g_optim], feed_dict={gan.z: z_batch, gan.x: np.zeros(z_batch.shape)})

                avg_loss += loss_D
                count += 1
            avg_loss /= count
            z = np.random.normal(size=(100, 2))
            excerpt = np.random.randint(1000, size=100)
            fake_x, real_logits, fake_logits = sess.run([gan.fake_x, gan.real_logits, gan.fake_logits],
                                                        feed_dict={gan.z: z, gan.x: X[excerpt, :]})
            accuracy = 0.5 * (np.sum(real_logits > 0) / 100. + np.sum(fake_logits < 0) / 100.)
            print('\ndiscriminator loss at epoch %d: %f' % (epoch, avg_loss))
            print('\ndiscriminator accuracy at epoch %d: %f' % (epoch, accuracy))
            plt.scatter(X[:, 0], X[:, 1])
            plt.scatter(fake_x[:, 0], fake_x[:, 1])
            plt.show()
        # writer = tf.summary.FileWriter('./logs', sess.graph)
        # writer.close()