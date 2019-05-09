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


def generator(z, output_dim, n_unit=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_unit)
        h1 = tf.nn.leaky_relu(h1, alpha)
        logits = tf.layers.dense(h1, output_dim)
        out = tf.nn.tanh(logits)
        return logits, out


def discriminator(x, n_unit=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_unit)
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
    g_logits, g_model = generator(input_z, input_size, g_hidden, reuse=False, alpha=alpha)
    d_logits_real, d_model_real = discriminator(input_real, d_hidden, reuse=False, alpha=alpha)
    d_logits_fake, d_model_fake = discriminator(g_model, d_hidden, reuse=True, alpha=alpha)

    d_label_real = tf.ones_like(d_logits_real) * (1 - smooth)
    d_label_fake = tf.zeros_like(d_logits_fake)

    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=d_label_real)
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=d_label_fake)

    d_loss = tf.reduce_mean(d_loss_fake + d_logits_real)

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

    learning_rate = 0.01

    total_var = tf.trainable_variables()
    d_var = [v for v in total_var if v.name.startswith('discriminator')]
    g_var = [v for v in total_var if v.name.startswith('generator')]

    d_train_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_var)
    g_train_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_var)

    batch_size = 100
    epochs = 1
    samples = []
    losses = []
    saver = tf.train.Saver(var_list=g_var)  # save weights
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for _ in range(mnist.train.num_examples // batch_size):
                batch = mnist.train.next_batch(batch_size)

                batch_image = batch[0].reshape((batch_size, 784))
                batch_image = batch_image * 2 - 1  # (0~1) to (-1, 1)

                batch_z = np.random.uniform(low=-1, high=1, size=(batch_size, z_size))

                sess.run(d_train_opt, feed_dict={input_real: batch_image, input_z: batch_z})  # G(z) as input of D
                sess.run(g_train_opt, feed_dict={input_z: batch_z})  # generate G(z)
                # weights will only be updated when running optimizer

            train_loss_d = sess.run(d_loss, feed_dict={input_real: batch_image, input_z: batch_z})
            train_loss_g = g_loss.eval(
                {input_z: batch_z})  # t.eval() is a shortcut for calling tf.get_default_session().run(t)
            print('Epoch {}/{} ...'.format(epoch + 1, epochs),
                  'd_loss: {:.4f} ...'.format(train_loss_d),
                  'g_loss: {:.4f} ...'.format(train_loss_g))
            losses.append((d_loss, g_loss))

            sample_z = np.random.uniform(-1, 1, size=(16, z_size))
            gen_samples = sess.run(
                generator(input_z, input_size, reuse=True),
                feed_dict={input_z: sample_z})
            samples.append(gen_samples)
            saver.save(sess, './checkpoints/generator.ckpt', global_step=epoch)
