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
from tensorflow.python.ops.gradients import AggregationMethod


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    DEFAULTS = {
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "KL_loss_coeff": 1,
        "samples_per_batch": 1
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, build_dict=None, d_hyperparams={}, scope='VAE',
                 save_graph_def=True, log_dir="./log", model_to_restore=False):
        """(Re)build a symmetric VAE model with given:
            * build_dict (if the model is being built new. The dict should contain the following keys:
                * encoder (callable object that takes input tensor as argument and returns tensors z_mean, z_log_sigma
                * decoder (callable object that takes z as input and returns reconstructed x)
                * input_size (number of inputs at each time step)
                * input_placeholder (placeholder object for inputs)
                * latent_size (dimension of latent (z) space)
                * dataset (DatasetFeed object for training)
            * d_hyperparameters (optional dictionary of updates to `DEFAULTS`)
            * model_to_restore (filename of model to generate from (provide filename, without .meta)
        """
        self.sess = tf.Session()

        if build_dict:
            assert not model_to_restore
            print(build_dict)
            assert 'dataset' in build_dict
            assert all(key in build_dict for key in ['encoder', 'decoder', 'n_input',
                                                     'input_placeholder', 'latent_dim',
                                                     'dataset', 'model_folder'])
            self.encoder = build_dict['encoder']
            self.decoder = build_dict['decoder']
            self.input_placeholder = build_dict['input_placeholder']
            self.shifted_input_placeholder = build_dict['shifted_input_placeholder']
            self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
            self.n_input = build_dict['n_input']
            self.latent_dim = build_dict['latent_dim']
            self.dataset = build_dict['dataset']
            self.model_folder = build_dict['model_folder']
            self.batch_size = self.dataset.minibatch_size
            # build graph
            self.scope = scope
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sess.run(tf.initialize_all_variables())
        elif model_to_restore:
            assert not build_dict
            model_datetime, model_name = os.path.basename(model_to_restore).split("_vae_")  # basename gives just the filename
            self.datetime = "{}_reloaded".format(model_datetime)
            *model_architecture, _ = re.split("_|-", model_name)
            # rebuild graph
            meta_graph = os.path.abspath(model_to_restore)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sess, meta_graph)
            handles = self.sess.graph.get_collection(VAE.RESTORE_KEY)
        else:
            raise Exception("VAE must be initialised with either build_dict or model_to_restore")

        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        # unpack handles for tensor ops to feed or fetch
        (self.z_mean, self.z_log_sigma, self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.kl_loss, self.rec_loss, self.l2_reg, self.global_step, self.train_op) = handles

        if save_graph_def: # tensorboard
            self.logger = tf.train.SummaryWriter(log_dir, self.sess.graph)

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sess)

    def save_model(self, outdir):
        """Saves the model if a self.saver object exists"""
        try:
            outfile = outdir + 'model'
            self.saver.save(self.sess, outfile, global_step=self.step)
        except AttributeError:
            return

    def _buildGraph(self):
        with tf.variable_scope(self.scope) as graph_scope:
            # Both will have size (n_steps, batch_size * samples_per_batch, n_inputs)
            x_in = self.input_placeholder
            x_shifted = self.shifted_input_placeholder
            assert x_in.get_shape()[1] % self.samples_per_batch == 0
            assert x_shifted.get_shape()[1] % self.samples_per_batch == 0

            n_steps = x_in.get_shape()[0].value  # To convert to int (otherwise this returns Dimension object)

            print("x_in, x_shifted shape: ", x_in.get_shape(), x_shifted.get_shape())

            z_mean, z_log_sigma = self.encoder(x_in)  # (batch_size * samples_per_batch, latent_dim)

            print("Finished setting up encoder")
            print([var._variable for var in tf.all_variables()])

            print('z_mean shape: ', z_mean.get_shape())
            print('z_log_sigma shape: ', z_log_sigma.get_shape())

            # z is random variables with shape (batch_size * samples_per_batch, latent_dim)
            z = self.sampleGaussian(z_mean, z_log_sigma)  # Tensor z evaluates to different samples each time

            # decoding / "generative": p(x|z)
            # x_reconstructed is (n_steps, batch_size * samples_per_batch, n_outputs)
            x_reconstructed = self.decoder(z, inputs=x_shifted)  # Feed the ground truth to the decoder
            # x_reconstructed = self.decoder(z_mean, inputs=x_shifted)  # When KL cost is eliminated, can just use the mean

            # Reshape x_reconstructed into (n_steps, samples_per_batch, batch_size, n_inputs)
            x_reconstructed = tf.reshape(x_reconstructed, [n_steps, self.samples_per_batch, self.batch_size, self.n_input])
            x_in_reshaped = tf.reshape(x_in, [n_steps, self.samples_per_batch, self.batch_size, self.n_input])

            print("Finished setting up decoder")
            print([var._variable for var in tf.all_variables()])

            # reconstruction loss: mismatch b/w x & x_reconstructed
            # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis

            print('x_reconstructed shape: ', x_reconstructed.get_shape())  # (n_steps, sample_per_batch, batch_size, n_outputs)
            print('x_in_reshaped shape: ', x_in_reshaped.get_shape())  # (n_steps, sample_per_batch, batch_size, n_outputs)

            # rec_loss = VAE.crossEntropy(x_reconstructed, x_in)
            rec_loss = tf.reduce_mean((x_reconstructed - x_in_reshaped)**2, reduction_indices=[0, 1, 3])  # Reduce everything but the batch_size
            print('rec_loss shape:', rec_loss.get_shape())

            # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
            # Recall z_mean, z_log_sigma have shape (batch_size * samples_per_batch, latent_dim),
            # i.e. they are repeated (samples_per_batch) times
            # For the kl_loss we only want the means and stdevs over the batch_size
            z_mean_batch = z_mean[:self.batch_size, :]
            z_log_sigma_batch = z_log_sigma[:self.batch_size, :]
            kl_loss = VAE.kullbackLeibler(z_mean_batch, z_log_sigma_batch)
            print('kl_loss shape:', kl_loss.get_shape())

            with tf.name_scope("l2_regularization"):
                regularizers = [tf.nn.l2_loss(var) for var in self.sess.graph.get_collection(
                    "trainable_variables") if "weights" in var.name]
                l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

            with tf.name_scope("cost"):
                # average over minibatch
                cost = tf.reduce_mean(rec_loss + self.KL_loss_coeff * kl_loss, name="vae_cost")
                cost += l2_reg

            print("Defined loss functions")

            # optimization
            global_step = tf.Variable(0, trainable=False)
            with tf.name_scope("Adam_optimizer"):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                tvars = tf.trainable_variables()
                # Set AggregationMethod to try to avoid crash when computing all gradients simultaneously
                grads_and_vars = optimizer.compute_gradients(cost, tvars,
                                                             aggregation_method=AggregationMethod.EXPERIMENTAL_TREE)
                clipped = [(tf.clip_by_value(grad, -5, 5), tvar) # gradient clipping
                        for grad, tvar in grads_and_vars]
                train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                     name="minimize_cost")

            print("Defined training ops")

            print([var._variable for var in tf.all_variables()])

            # ops to directly explore latent space
            # defaults to prior z ~ N(0, I)
            with tf.name_scope("latent_in"):
                z_ = tf.placeholder_with_default(tf.random_normal([1, self.latent_dim]),
                                                 shape=[1, self.latent_dim],
                                                 name="latent_in")
            graph_scope.reuse_variables()  # No new variables should be created from this point on
            x_reconstructed_ = self.decoder(z_)

            return (z_mean, z_log_sigma, x_reconstructed,  # Removed dropout from second place
                    z_, x_reconstructed_, cost, kl_loss, rec_loss, l2_reg, global_step, train_op)

    def sampleGaussian(self, mu, log_sigma):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma)  # N(mu, I * sigma**2)

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
        return self.sess.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.sess.run(zs) if is_tensor(zs) else zs) # coerce to np.array
            feed_dict.update({self.z_: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sess.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def train(self, max_iter=np.inf, max_epochs=np.inf,
              verbose=True, save=True):
        if save:
            self.saver = tf.train.Saver(tf.all_variables())

        outdir = self.model_folder
        self.accumulated_cost = 0
        now = datetime.now().isoformat()[11:]
        print("------- Training begin: {} -------\n".format(now))

        while True:
            try:
                x = self.dataset.next_batch()  # (batch_size, n_steps, n_inputs)
                x = np.concatenate(tuple([x] * self.samples_per_batch), axis=0)
                x = np.transpose(x, [1, 0, 2])  # (n_steps, samples_per_batch * batch_size, n_inputs)

                total_batch_size = x.shape[1]  # batch_size * samples_per_batch
                n_inputs = x.shape[2]
                x_shifted = np.concatenate((np.zeros((1, total_batch_size, n_inputs)), x[0:-1, :, :]), axis=0)

                assert x.shape == x_shifted.shape

                feed_dict = {self.input_placeholder: x, self.shifted_input_placeholder: x_shifted}
                fetches = [self.x_reconstructed, self.cost, self.kl_loss, self.rec_loss, self.global_step, self.train_op]
                x_reconstructed, cost, kl_loss, rec_loss, i, _ = self.sess.run(fetches, feed_dict=feed_dict)

                self.accumulated_cost += cost

                if i%10 == 0 and verbose:
                    print("Step {}-> avg total cost: {}, cost for this minibatch: {}".format(i, self.accumulated_cost / i, cost))
                    print("   minibatch KL_cost = {}, reconst = {}".format(np.mean(kl_loss),
                                                                           np.mean(rec_loss)))
                if i % 500 == 0:
                    self.save_model(outdir)

                if i >= max_iter or self.dataset.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, self.dataset.epochs_completed, self.accumulated_cost / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    self.save_model(outdir)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError): # not logging
                        pass
                    return cost

            except(KeyboardInterrupt):
                print("\nTraining temporarily paused, enter an option:")
                print("1 - Terminate training")
                print("2 - Change learning rate (current is {})".format(self.learning_rate))
                print("3 - Change KL coeff (current is {})".format(self.KL_loss_coeff))
                option = None
                while not option in ['1', '2', '3']:
                    option = input()
                    if option == '1':
                        print("final avg cost (@ step {} = epoch {}): {}".format(
                            i, self.dataset.epochs_completed, self.accumulated_cost / i))
                        now = datetime.now().isoformat()[11:]
                        self.save_model(outdir)
                        print("------- Training end: {} -------\n".format(now))
                        return cost
                    elif option == '2':
                        new_learning_rate = None
                        while not type(new_learning_rate) == float:
                            new_learning_rate = input("Enter new learning rate: ")
                        self.learning_rate = new_learning_rate
                    elif option == '3':
                        new_kl_coeff = None
                        while not type(new_kl_coeff) == float:
                            new_kl_coeff = input("Enter new KL coeff: ")
                        self.KL_loss_coeff = new_kl_coeff
