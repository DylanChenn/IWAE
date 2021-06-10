from IWAE import IWAE
import numpy as np
import warnings
from tensorflow.keras.datasets import mnist
import os
import tensorflow as tf

warnings.filterwarnings("ignore")

a = tf.config.threading.get_inter_op_parallelism_threads()
b = tf.config.threading.get_intra_op_parallelism_threads()
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

np.random.seed(123)
tf.random.set_seed(123)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, newshape=(-1, 784))
x_test = np.reshape(x_test, newshape=(-1, 784))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
trainx = np.random.binomial(1, x_train, size=x_train.shape).astype('float32')
testx = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')

x_dim = trainx.shape[1]
z_dim = 50
h_dim = 200
n_sample = 1
batch_size = 20
model = IWAE(h_dim, z_dim)
count = int(trainx.shape[0]/batch_size)
model.train(trainx, n_sample, testx, batch_size, count)
