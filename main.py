import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))