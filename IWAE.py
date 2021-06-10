"""
using tensorflow to reproduce IWAE
tensorflow 2.5.0, tensorflow_probability 0.12.2
"""
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import time

np.random.seed(123)
tf.random.set_seed(123)


def get_bias():
    # ---- For initializing the bias in the final Bernoulli layer for p(x|z)
    (Xtrain, ytrain), (_, _) = tf.keras.datasets.mnist.load_data()
    Ntrain = Xtrain.shape[0]

    # ---- reshape to vectors
    Xtrain = Xtrain.reshape(Ntrain, -1) / 255

    train_mean = np.mean(Xtrain, axis=0)

    bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

    return tf.constant_initializer(bias)


class Encoder(tf.keras.Model):
    def __init__(self, h_dim, z_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.h1 = tf.keras.layers.Dense(h_dim, activation=tf.nn.tanh)
        self.h2 = tf.keras.layers.Dense(h_dim, activation=tf.nn.tanh)
        self.mu = tf.keras.layers.Dense(z_dim, activation=None)
        self.logvar = tf.keras.layers.Dense(z_dim, activation=None)

    def call(self, x, n_sample):
        h1 = self.h1(x)
        h2 = self.h2(h1)
        mu = self.mu(h2)
        logvar = self.logvar(h2)

        q_z_given_x = tfd.Normal(mu, tf.exp(logvar) + 1e-6)

        z = q_z_given_x.sample(n_sample)

        return z, q_z_given_x


class Decoder(tf.keras.Model):
    def __init__(self, h_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.h1 = tf.keras.layers.Dense(h_dim, activation=tf.nn.tanh)
        self.h2 = tf.keras.layers.Dense(h_dim, activation=tf.nn.tanh)
        self.logit = tf.keras.layers.Dense(784, activation=None, bias_initializer=get_bias())

    def call(self, z):
        h1 = self.h1(z)
        h2 = self.h2(h1)
        logits = self.logit(h2)

        p_x_given_z = tfd.Bernoulli(logits=logits)

        return logits, p_x_given_z


class IWAE(tf.keras.Model):
    def __init__(self, h_dim, z_dim, **kwargs):
        super(IWAE, self).__init__(**kwargs)

        self.encoder = Encoder(h_dim, z_dim)
        self.decoder = Decoder(h_dim)

        self.lr = {}
        self.epochs = 0
        for i in range(8):
            self.lr[self.epochs] = 0.001 * 10 ** (-i / 7)
            self.epochs += 3 ** i

    def call(self, x, n_sample):
        z, qzx = self.encoder(x, n_sample)  # z.shape = n_sample*batch*z_dim
        logits, pxz = self.decoder(z)

        log_qzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)  # n_sample*batch
        log_pxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)  # n_sample*batch
        pz = tfd.Normal(0, 1)
        log_pz = tf.reduce_sum(pz.log_prob(z), axis=-1)  # n_sample*batch

        log_w = log_pxz + log_pz - log_qzx  # n_sample*batch
        max_log_w = tf.reduce_max(log_w, axis=0)  # batch
        iwae_elbo = tf.reduce_mean(
            tf.math.log(tf.reduce_mean(tf.exp(log_w - max_log_w), axis=0))+max_log_w
        )
        return iwae_elbo

    def train_step(self, x, n_samples, optimizer):
        with tf.GradientTape() as tape:
            res = -self.call(x, n_samples)

        grads = tape.gradient(res, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return res

    def train(self, trainx, n_sample, x_test, batch_size, count):
        optimizer = tf.keras.optimizers.Adam(self.lr[0], epsilon=1e-4)
        start = time.time()
        for epoch in range(self.epochs):
            x = (tf.data.Dataset.from_tensor_slices(trainx).shuffle(trainx.shape[0]).batch(batch_size))
            if epoch in self.lr:
                print("Changing learning rate from {0} to {1}".format(
                    optimizer.learning_rate.numpy(), self.lr[epoch])
                )
                optimizer.learning_rate.assign(self.lr[epoch])
            for it, x_batch in enumerate(x):
                s = time.time()
                with tf.GradientTape() as tape:
                    loss = -self.call(x_batch, n_sample)
                    grads = tape.gradient(loss, self.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.trainable_weights))
                if it % 200 == 0:
                    test_loss = -self.eval_test(x_test, n_sample)
                    took = time.time() - start
                    start = time.time()
                    print(
                        "Epoch {:}/{:}\t{:}/{:}\t\tloss:{:.4f}\ttestloss:{:.4f}\ttime:{:.2f}".format(
                            epoch + 1, int(self.epochs), it, count, loss, test_loss, took)
                    )
            test_loss = -self.eval_test(x_test, n_sample)
            print("Epoch {:}/{:}\ttestloss: {:.4f}".format(epoch + 1, self.epochs, test_loss))
        nll = self.NLL_test(x_test, 5000)
        print("NLLTest: {:.4f} with k={:}".format(nll, n_sample))

    def eval_test(self, x_test, n_sample):
        return self.call(x_test, n_sample)

    def NLL_test(self, x_test, n_sample):
        iwae_elbo = 0
        count = x_test.shape[0]
        for i, x in enumerate(x_test):
            iwae_elbo += -self.call(np.expand_dims(x, axis=0), n_sample)
            print("Test Sample {:}/{:}".format(i+1, count))
        return iwae_elbo/x_test.shape[0]
