from datetime import datetime
import os
import re
import sys
import json
import numpy as np
import tensorflow as tf

import utils.vaeplot as vaeplot
# from utils.utils import print_
from utils.training_utils import setup_training_ops


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    DEFAULTS = {
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "samples_per_batch": 1,
        "max_batch_size_for_gradients": None,
        "deterministic_warm_up": 0
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, build_dict=None, d_hyperparams={}, scope='VAE',
                 save_graph_def=True, log_dir="./log/", analysis_dir="./analysis/", model_to_restore=False, json_dict=None):
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
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.analysis_folder = analysis_dir

        if build_dict:
            assert not model_to_restore
            assert 'dataset' in build_dict
            assert all(key in build_dict for key in ['encoder', 'decoder', 'n_input',
                                                     'input_placeholder', 'latent_dim',
                                                     'dataset', 'model_folder'])
            self.encoder = build_dict['encoder']
            self.decoder = build_dict['decoder']
            self.input_placeholder = build_dict['input_placeholder']
            self.shifted_input_placeholder = build_dict['shifted_input_placeholder']

            self.n_input = build_dict['n_input']
            self.latent_dim = build_dict['latent_dim']
            self.dataset = build_dict['dataset']
            self.model_folder = build_dict['model_folder']
            self.batch_size = self.dataset.minibatch_size
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.json_dict = json_dict
            self.datetime = json_dict['model_datetime']
            # build graph
            self.scope = scope
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sess.run(tf.initialize_all_variables())

            # unpack handles for tensor ops to feed or fetch
            (_1, _2,
             self.z_mean, self.z_log_sigma, self.x_reconstructed, self.z_, self.x_reconstructed_,
             self.cost, self.cost_no_KL, self.kl_loss, self.rec_loss, self.l2_reg,
             self.apply_gradients_op, self.apply_gradients_op_no_KL, _3) = handles
        elif model_to_restore:
            assert not build_dict
            self.model_folder = '/'.join((model_to_restore.split('/')[:-1])) + '/'
            with open(self.model_folder + '/network_settings.json') as network_json_file:
                json_vector_settings_dict = json.load(network_json_file)
            model_datetime = json_vector_settings_dict['model_datetime']
            self.datetime = "{}_reloaded".format(model_datetime)

            # rebuild graph
            meta_graph = os.path.abspath(model_to_restore)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.sess, meta_graph)

            handles = self.sess.graph.get_collection(VAE.RESTORE_KEY)
            print("Restored handles: ", handles)
            (self.input_placeholder, self.shifted_input_placeholder,
             self.z_mean, self.z_log_sigma, self.x_reconstructed, self.z_, self.x_reconstructed_,
             self.cost, self.cost_no_KL, self.kl_loss, self.rec_loss, self.l2_reg,
             self.apply_gradients_op, self.apply_gradients_op_no_KL, self.global_step) = handles

            self.optimizer, self.gradient_acc, self.gradient_acc_no_KL, apply_gradients_op, apply_gradients_op_no_KL = \
                setup_training_ops(self.learning_rate, self.cost, self.cost_no_KL, self.global_step)
        else:
            raise Exception("VAE must be initialised with either build_dict or model_to_restore")

        if save_graph_def:  # tensorboard
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
            print("Failed to save model at step {}".format(self.step))
            return

    def _buildGraph(self):
        with tf.variable_scope(self.scope) as graph_scope:
            # Both will have size (n_steps, batch_size * samples_per_batch, n_inputs)
            assert self.input_placeholder.get_shape()[1] % self.samples_per_batch == 0
            assert self.shifted_input_placeholder.get_shape()[1] % self.samples_per_batch == 0

            n_steps = self.input_placeholder.get_shape()[0].value  # To convert to int (otherwise this returns Dimension object)

            print("input, shifted_input shape: ", self.input_placeholder.get_shape(),
                  self.shifted_input_placeholder.get_shape())

            z_mean, z_log_sigma = self.encoder(self.input_placeholder)  # (batch_size * samples_per_batch, latent_dim)

            print("Finished setting up encoder")
            print([var._variable for var in tf.all_variables()])

            print('z_mean shape: ', z_mean.get_shape())
            print('z_log_sigma shape: ', z_log_sigma.get_shape())

            # z is random variables with shape (batch_size * samples_per_batch, latent_dim)
            z = self.sampleGaussian(z_mean, z_log_sigma)  # Tensor z evaluates to different samples each time

            # decoding / "generative": p(x|z)
            # reconstruction is (n_steps, batch_size * samples_per_batch, n_outputs)
            reconstruction = self.decoder(z, inputs=self.shifted_input_placeholder)  # Feed the ground truth to the decoder
            # reconstruction = self.decoder(z_mean, inputs=x_shifted)  # When KL cost is eliminated, can just use the mean

            # Reshape reconstruction into (n_steps, samples_per_batch, batch_size, n_inputs)
            reconstruction = tf.reshape(reconstruction, [n_steps, self.samples_per_batch, self.batch_size, self.n_input])
            input_reshaped = tf.reshape(self.input_placeholder, [n_steps, self.samples_per_batch, self.batch_size, self.n_input])

            print("Finished setting up decoder")
            print([var._variable for var in tf.all_variables()])

            # reconstruction loss: mismatch b/w x & reconstruction
            # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis

            print('reconstruction shape: ', reconstruction.get_shape())  # (n_steps, sample_per_batch, batch_size, n_outputs)
            print('input_reshaped shape: ', input_reshaped.get_shape())  # (n_steps, sample_per_batch, batch_size, n_outputs)

            # rec_loss = VAE.crossEntropy(reconstruction, x_in)
            rec_loss = tf.reduce_mean((reconstruction - input_reshaped)**2, reduction_indices=[0, 1, 3])  # Reduce everything but the batch_size
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

            #TODO: Apply data masks?
            with tf.name_scope("cost"):
                # average over minibatch
                cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
                cost += l2_reg

            with tf.name_scope("cost_no_KL"):
                cost_no_KL = tf.reduce_mean(rec_loss)
                cost_no_KL += l2_reg

            print("Defined loss functions")

            # optimization
            self.optimizer, self.gradient_acc, self.gradient_acc_no_KL, apply_gradients_op, apply_gradients_op_no_KL = \
                setup_training_ops(self.learning_rate, cost, cost_no_KL, self.global_step)
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

            return (self.input_placeholder, self.shifted_input_placeholder, z_mean, z_log_sigma,
                    reconstruction,  # Removed dropout from second place
                    z_, x_reconstructed_, cost, cost_no_KL, kl_loss, rec_loss, l2_reg,
                    apply_gradients_op, apply_gradients_op_no_KL, self.global_step)

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
        feed_dict = {self.input_placeholder: x}
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

    def train(self, max_iter=np.inf, max_epochs=np.inf, verbose=True, save=True):

        # Get ops for gradient updates
        update_gradients_ops = self.gradient_acc.update_gradients_ops()
        clear_gradients_ops = self.gradient_acc.clear_gradients()

        update_gradients_ops_no_KL = self.gradient_acc_no_KL.update_gradients_ops()
        clear_gradients_ops_no_KL = self.gradient_acc_no_KL.clear_gradients()

        # # TEMPORARY HACK - shouldn't be needed now that the saver is initialized after the gradient ops are defined
        # uninitialized_vars = []
        # for var in tf.all_variables():
        #     try:
        #         self.sess.run(var)
        #     except tf.errors.FailedPreconditionError:
        #         uninitialized_vars.append(var)
        #
        # self.sess.run(tf.initialize_variables(uninitialized_vars))

        if save:
            self.saver = tf.train.Saver(tf.all_variables())

        total_cost_history = np.array([])
        KL_cost_history = np.array([])
        reconstruction_cost_history = np.array([])

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

                # Reverse the input to the encoder in time!
                x = x[::-1, :, :]
                assert x[10, 10, 10] == x_shifted[-10, 10, 10]
                assert x[4, 24, 2] == x_shifted[-4, 24, 2]

                assert x.shape == x_shifted.shape

                if self.max_batch_size_for_gradients and total_batch_size > self.max_batch_size_for_gradients:
                    pass

                feed_dict = {self.input_placeholder: x, self.shifted_input_placeholder: x_shifted}

                print("Updating gradients...")
                if self.deterministic_warm_up:
                    fetches = [self.x_reconstructed, self.cost_no_KL, self.kl_loss, self.rec_loss, self.global_step] + \
                              update_gradients_ops_no_KL
                else:
                    fetches = [self.x_reconstructed, self.cost, self.kl_loss, self.rec_loss, self.global_step] + \
                              update_gradients_ops
                x_reconstructed, cost, kl_loss, rec_loss, i, *_ = self.sess.run(fetches, feed_dict=feed_dict)

                # print("Gradients before being cleared:")
                # print([(k, self.sess.run(self.gradient_acc._var_to_accum_grad[k], feed_dict=feed_dict))
                #        for k in self.gradient_acc._var_to_accum_grad])

                print("Applying gradients...", end='')
                if self.deterministic_warm_up:
                    self.sess.run(self.apply_gradients_op_no_KL, feed_dict=feed_dict)
                    print("done")
                    self.deterministic_warm_up -= 1
                    print("{} more steps of zero-KL cost".format(self.deterministic_warm_up))
                    self.sess.run(clear_gradients_ops_no_KL)
                else:
                    self.sess.run(self.apply_gradients_op, feed_dict=feed_dict)
                    print("done")
                    self.sess.run(clear_gradients_ops)

                print("Step {}-> cost for this minibatch: {}".format(i, cost))
                print("   minibatch KL_cost = {}, reconst = {}".format(np.mean(kl_loss), np.mean(rec_loss)))

                # print("Gradients should be zero: ")
                # print([(k, self.sess.run(self.gradient_acc._var_to_accum_grad[k], feed_dict=feed_dict))
                #        for k in self.gradient_acc._var_to_accum_grad])

                # if i % 10 == 0 and verbose:
                #     print("Step {}-> cost for this minibatch: {}".format(i, cost))
                #     print("   minibatch KL_cost = {}, reconst = {}".format(np.mean(kl_loss),
                #                                                            np.mean(rec_loss)))

                total_cost_history = np.hstack((total_cost_history, np.array([float(cost)])))
                KL_cost_history = np.hstack((KL_cost_history, np.array([float(kl_loss)])))
                reconstruction_cost_history = np.hstack((reconstruction_cost_history, np.array([float(rec_loss)])))

                if i % 250 == 0:
                    # SAVE ALL MODEL PARAMETERS
                    self.save_model(outdir)

                if i % 5 == 0:
                    # SAVE COSTS FOR LEARNING CURVES
                    np.save(self.analysis_folder + 'total_cost.npy', total_cost_history)
                    np.save(self.analysis_folder + 'KL_cost.npy', KL_cost_history)
                    np.save(self.analysis_folder + 'reconstruction_cost.npy', reconstruction_cost_history)

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
