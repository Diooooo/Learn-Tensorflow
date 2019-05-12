import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generator(X):
    with tf.variable_scope('generator'):
        l = tf.layers.dense(X, 256, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        l = tf.layers.dense(l, 512, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        l = tf.layers.dense(l, 1024, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        g_logits = tf.layers.dense(X, 784, activation=tf.nn.tanh,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='logits')
    return g_logits


def discriminator(Y, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        l = tf.layers.dense(Y, 1024, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        l = tf.layers.dropout(l)
        l = tf.layers.dense(l, 1024, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        l = tf.layers.dropout(l)
        l = tf.layers.dense(l, 512, activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        l = tf.layers.dropout(l)
        d_logits = tf.layers.dense(l, 1, activation=tf.nn.sigmoid,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='logits')
    return d_logits


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

    epochs = 20
    batch_size = 128

    X = tf.placeholder(tf.float32, [batch_size, 100])
    Y = tf.placeholder(tf.float32, [batch_size, 784])

    g_logits = generator(X)
    d_logits_real = discriminator(Y)
    d_logits_feak = discriminator(g_logits, reuse=True)

    eps = 1e-6
    g_loss = tf.reduce_mean(tf.log(d_logits_feak + eps))
    d_loss = tf.reduce_mean(tf.log(d_logits_real + eps) + tf.log(1 - d_logits_feak + eps))

    var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    var_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
    g_optimizer = tf.train.AdamOptimizer().minimize(-g_loss, var_list=var_g)
    d_optimizer = tf.train.AdamOptimizer().minimize(-d_loss, var_list=var_d)

    iter_per_epoch = int(np.ceil(mnist.train.num_examples // batch_size))
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            train_l = 0
            val_l = 0
            for i in range(iter_per_epoch):
                z = np.random.normal(0, 1, (batch_size, 100))
                y_train, _ = mnist.train.next_batch(batch_size)
                # y_val, _ = mnist.validation.next_batch(batch_size)
                sum_d = 0
                for _ in range(1):
                    _, loss_d = sess.run([d_optimizer, d_loss], feed_dict={X: z, Y: y_train})
                    sum_d += loss_d
                _, loss_g = sess.run([g_optimizer, g_loss], feed_dict={X: z})
                losses.append([-loss_g, -sum_d / 5])

            print('Epoch {:d}--g loss:{:f}, d loss:{:f}'.format(epoch + 1, loss_g,
                                                                loss_d))
    losses = np.asarray(losses)

    plt.plot(losses[:, 0])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.savefig('./mnist_g.png')
    plt.show()

    plt.plot(losses[:, 1])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.savefig('./mnist_d.png')
    plt.show()
