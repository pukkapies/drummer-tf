from datetime import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf

import utils.vaeplot as vaeplot
from utils.functionaltools import composeAll
from nn_models.layers import Dense, FeedForward
# from utils.utils import print_


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    DEFAULTS = {
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, encoder, decoder, input_size, input_placeholder, latent_size, dataset, d_hyperparams={},
                 save_graph_def=True, log_dir="./log"):
        """(Re)build a symmetric VAE model with given:

            * encoder (callable object that takes input tensor as argument and returns tensors z_mean, z_log_sigma

            * decoder (callable object that takes z as input and returns reconstructed x)

            *

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """
        self.encoder = encoder
        self.decoder = decoder
        self.input_placeholder = input_placeholder
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.input_size = input_size
        self.latent_size = latent_size
        self.sesh = tf.Session()
        self.dataset = dataset
        self.batch_size = self.dataset.minibatch_size

        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        # build graph
        handles = self._buildGraph()
        for handle in handles:
            tf.add_to_collection(VAE.RESTORE_KEY, handle)
        self.sesh.run(tf.initialize_all_variables())

        # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.global_step, self.train_op) = handles

        if save_graph_def: # tensorboard
            self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = self.input_placeholder

        z_mean, z_log_sigma = self.encoder(x_in)

        print('z_mean shape: ', z_mean.get_shape())
        print('z_log_sigma shape: ', z_log_sigma.get_shape())

        # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
        z = self.sampleGaussian(z_mean, z_log_sigma)

        # decoding / "generative": p(x|z)
        x_reconstructed = self.decoder(z)

        print([var._variable for var in tf.all_variables()])

        # reconstruction loss: mismatch b/w x & x_reconstructed
        # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis

        print('x_reconstructed shape: ', x_reconstructed.get_shape())
        print('x_in shape: ', x_in.get_shape())

        # rec_loss = VAE.crossEntropy(x_reconstructed, x_in)
        rec_loss = tf.reduce_mean((x_reconstructed - x_in)**2)

        # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
        kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma)

        print("Defined loss functions")

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection(
                "trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # average over minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            cost += l2_reg

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) # gradient clipping
                    for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name="minimize_cost")

        print("Defined training ops")

        print([var._variable for var in tf.all_variables()])

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)
        with tf.name_scope("latent_in"):
            z_ = tf.placeholder_with_default(tf.random_normal([1, self.latent_size]),
                                            shape=[1, self.latent_size],
                                            name="latent_in")
        x_reconstructed_ = self.decoder(z_)

        return (x_in, z_mean, z_log_sigma, x_reconstructed,  # Removed dropout from second place
                z_, x_reconstructed_, cost, global_step, train_op)

    def sampleGaussian(self, mu, log_sigma):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

    @staticmethod
    def crossEntropy(obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)

    @staticmethod
    def l1_loss(obs, actual):
        """L1 loss (a.k.a. LAD), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l1_loss"):
            return tf.reduce_sum(tf.abs(obs - actual) , 1)

    @staticmethod
    def l2_loss(obs, actual):
        """L2 loss (a.k.a. Euclidean / LSE), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l2_loss"):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 -
                                        tf.exp(2 * log_sigma), 1)

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.sesh.run(zs) if is_tensor(zs) else zs) # coerce to np.array
            feed_dict.update({self.z_: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sesh.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def train(self, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
              verbose=True, save=True, outdir="./out", plots_outdir="./png",
              plot_latent_over_time=False):
        print("inside training function")
        if save:
            saver = tf.train.Saver(tf.all_variables())

        try:
            err_train = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            while True:
                x = self.dataset.next_batch()  # (batch_size, n_steps, n_inputs)
                print("Next data batch shape: ", x.shape)
                x = np.transpose(x, [1, 0, 2])  # (n_steps, batch_size, n_inputs)
                print("After transposing: ", x.shape)

                feed_dict = {self.x_in: x}
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict=feed_dict)

                err_train += cost

                if i%10 == 0 and verbose:
                    print("round {} --> avg cost: ".format(i), err_train / i)

                # if i%2000 == 0 and verbose:# and i >= 10000:
                    # if cross_validate:
                    #     x, _ = X.validation.next_batch(self.batch_size)
                    #     feed_dict = {self.x_in: x}
                    #     fetches = [self.x_reconstructed, self.cost]
                    #     x_reconstructed, cost = self.sesh.run(fetches, feed_dict)
                    #
                    #     print("round {} --> CV cost: ".format(i), cost)
                    #     vaeplot.plotSubset(self, x, x_reconstructed, n=10, name="cv",
                    #                     outdir=plots_outdir)

                if i % 500 == 0:
                    outfile = os.path.join(os.path.abspath(outdir), "{}_vae_{}".format(
                        self.datetime, "_".join(map(str, 'LSTM'))))
                    saver.save(self.sesh, outfile, global_step=self.step)

                if i >= max_iter or self.dataset.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, self.dataset.epochs_completed, err_train / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir), "{}_vae_{}".format(
                            self.datetime, "_".join(map(str, 'LSTM'))))
                        saver.save(self.sesh, outfile, global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError): # not logging
                        continue
                    break

        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, self.dataset.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)

