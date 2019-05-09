import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

if __name__ == '__main__':
    test = np.load('./test_data_27.npy')
    test = test.item()
    # index = 1
    # sketch = cv2.imread('./results/sketch-{}.jpg'.format(index))
    conditions = []
    for i in range(20):
        # if i == 7:
        #     c = cv2.imread('./handdraw/{}.png'.format(i))
        # else:
        #     c = cv2.imread('./handdraw/{}.jpg'.format(i))
        c = cv2.imread('./results/condition-{}.jpg'.format(i + 1))
        c = gaussian_filter(c, 20)
        conditions.append(c)
    conditions = np.asarray(conditions)

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('./unet31/checkpoint-50.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./unet31'))
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('input:0')
    condition = graph.get_tensor_by_name('condition:0')
    is_train = graph.get_tensor_by_name('is_training:0')
    logits = graph.get_collection('g_logits')

    generated = sess.run(logits, feed_dict={inputs: test['input'],
                                            condition: conditions,
                                            is_train: False})

    plt.figure()
    for index in range(20):
        pred = generated[0][index]
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 5, index + 1)
        plt.imshow(1 - pred)
    plt.show()
    # for index in range(20):
    #     plt.figure()
    #     plt.subplot(4, 5, 1)
    #     target = test['input'][index].reshape(256, 256)
    #     # target = cv2.cvtColor(1 - target, cv2.COLOR_BGR2RGB)
    #     plt.imshow(1 - target, cmap='gray')
    #     plt.subplot(4, 5, 2)
    #     target = test['condition'][index]
    #     target = cv2.cvtColor(1 - target, cv2.COLOR_BGR2RGB)
    #     plt.imshow(target)
    #     plt.subplot(4, 5, 3)
    #     target = test['target'][index]
    #     target = cv2.cvtColor(1 - target, cv2.COLOR_BGR2RGB)
    #     plt.imshow(target)
    #     for i in range(10):
    #         res = np.load('./gan/predicted_pre_30-{}.npy'.format(10 * (i + 1)))
    #         plt.subplot(4, 5, 6 + i)
    #         predicted = res[index]
    #         predicted = cv2.cvtColor(1 - predicted, cv2.COLOR_BGR2RGB)
    #         plt.imshow(predicted)
    #
    #     plt.show()
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(target)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(predicted)
    #     plt.show()
    # predicted = np.load('./wgangp/predicted_pre_31-50.npy')
    # for index in range(20):
    #     # sketch = 1 - test['input'][index]
    #     # sketch *= 255
    #     # cv2.imwrite('./results/sketch-{}.jpg'.format(index + 1), sketch)
    #     # condition = 1 - test['condition'][index]
    #     # condition *= 255
    #     # cv2.imwrite('./results/condition-{}.jpg'.format(index + 1), condition)
    #     # target = 1 - test['target'][index]
    #     # target *= 255
    #     # cv2.imwrite('./results/target-{}.jpg'.format(index + 1), target)
    #     pred = 1 - predicted[index]
    #     pred *= 255
    #     cv2.imwrite('./results/wgangp-{}.jpg'.format(index + 1), pred)
