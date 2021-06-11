from IWAE import IWAE
import numpy as np
import warnings
from tensorflow.keras.datasets import mnist
import time
import tensorflow as tf

warnings.filterwarnings("ignore")

np.random.seed(123)
tf.random.set_seed(123)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, newshape=(-1, 784))
x_test = np.reshape(x_test, newshape=(-1, 784))
trainx = x_train.astype('float32') / 255
testx = x_test.astype('float32') / 255
trainx = np.random.binomial(1, trainx, size=x_train.shape).astype('float32')
testx = np.random.binomial(1, testx, size=x_test.shape).astype('float32')
# trainx[trainx >= 0.5] = 1
# trainx[trainx < 0.5] = 0
# testx[testx >= 0.5] = 1
# testx[testx < 0.5] = 0

x_dim = trainx.shape[1]
z_dim = 50
h_dim = 200
n_sample = 1
batch_size = 20
model = IWAE(h_dim, z_dim)
count = int(trainx.shape[0]/batch_size)
lr, epochs = model.get_lr()

optimizer = tf.keras.optimizers.Adam(lr[0], epsilon=1e-4)
start = time.time()
full_time_start = time.time()
for epoch in range(epochs):
    x = (tf.data.Dataset.from_tensor_slices(trainx).shuffle(trainx.shape[0]).batch(batch_size))
    if epoch in lr:
        print("Changing learning rate from {0} to {1}".format(
            optimizer.learning_rate.numpy(), lr[epoch])
        )
        optimizer.learning_rate.assign(lr[epoch])
    for it, x_batch in enumerate(x):
        s = time.time()
        loss = model.train_step(x_batch, n_sample, optimizer)
        if (it+1) % 200 == 0:
            test_loss = -model.eval_test(testx, n_sample)
            t = time.time() - start
            start = time.time()
            print(
                "Epoch {:}/{:}\t{:}/{:}\t\tloss:{:.4f}\ttestloss:{:.4f}\ttime:{:.2f}".format(
                    epoch + 1, epochs, (it+1), count, loss, test_loss, t)
            )
    test_loss = -model.eval_test(testx, n_sample)
    full_time_past = time.time() - full_time_start
    full_time_past = full_time_past / (epoch+1) * (epochs - epoch - 1)
    print("Epoch {:}/{:}\ttestloss: {:.4f}\tetc time: {:}h{:}m{:}s".format(
        epoch + 1, epochs, test_loss, int(full_time_past/3600), int(full_time_past % 3600/60), int(full_time_past % 60)
    ))

iwae_elbo = 0
count = testx.shape[0]
for i, x in enumerate(testx):
    iwae_elbo += -model(np.expand_dims(x, axis=0), 5000)
    print("Test Sample {:}/{:}".format(i + 1, count))
nll = iwae_elbo / testx.shape[0]
print("NLLTest: {:.4f} with k={:}".format(nll, n_sample))
