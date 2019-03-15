import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def fc_layer(input, name, shape, use_norm=True):
    with tf.name_scope(name) as scope:
        W = tf.get_variable(name=name + '_W', shape=[shape[0], shape[1]],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        b = tf.get_variable(name=name + '_b', shape=[shape[1]], initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=True)
        if use_norm:
            x_norm = tf.layers.batch_normalization(tf.matmul(input, W) + b, training=True, name=name + '_BN')
        else:
            x_norm = tf.matmul(input, W) + b
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
    epochs = 10

    X = tf.placeholder(tf.float32, [batch_size, 784])
    Y = tf.placeholder(tf.float32, [batch_size, 10])

    layer1 = fc_layer(X, 'input', (784, 1024))
    layer2 = fc_layer(layer1, 'hidden1', (1024, 1024))
    layer3 = fc_layer(layer2, 'hidden2', (1024, 1024))
    logits = fc_layer(layer3, 'output', (1024, 10), use_norm=False)

    train_loss = loss_function(logits, Y, 'train')
    val_loss = loss_function(logits, Y, 'val')
    train = optimizer(learning_rate, train_loss, 'sgd')

    train_val_loss = []
    iter_per_epoch = int(np.ceil(mnist.train.num_examples // batch_size))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            train_l = 0
            val_l = 0
            for _ in range(iter_per_epoch):
                x_train, y_train = mnist.train.next_batch(batch_size)
                x_val, y_val = mnist.validation.next_batch(batch_size)
                _, loss_train = sess.run([train, train_loss], feed_dict={X: x_train, Y: y_train})
                loss_val = sess.run(val_loss, feed_dict={X: x_val, Y: y_val})
                train_val_loss.append([loss_train, loss_val])
                train_l += loss_train
                val_l += loss_val
            print('Epoch {:d}--train loss:{:f}, val loss:{:f}'.format(i+1, train_l / iter_per_epoch,
                                                                      val_l / iter_per_epoch))
    train_val_loss = np.asarray(train_val_loss)
    plt.plot(train_val_loss[:, 0], label='train loss')
    plt.plot(train_val_loss[:, 1], label='val loss')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('MNIST with FNN')
    plt.savefig('./mnist.jpg')
    plt.show()
