from datetime import datetime
import os
import re
import sys
import json
import numpy as np
import tensorflow as tf

import utils.vaeplot as vaeplot
# from utils.utils import print_
from tensorflow.python.ops import rnn_cell
from utils.training_utils import setup_AE_training_ops, TrainingLog


class Autoencoder():
    """Standard Autoencoder
    """
    DEFAULTS = {
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "max_batch_size_for_gradients": None,
        "num_batches_per_grad_update": 1
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, build_dict=None, d_hyperparams={}, scope='autoencoder',
                 save_graph_def=True, log_dir="./log/", analysis_dir="./analysis/", model_to_restore=False, json_dict=None):
        """(Re)build a symmetric VAE model with given:
            * build_dict (if the model is being built new. The dict should contain the following keys:
                * encoder (callable object that takes input tensor as argument and returns encoding
                * decoder (callable object that takes encoding as input and returns reconstructed x)
                * input_size (number of inputs at each time step)
                * input_placeholder (placeholder object for inputs)
                * latent_size (dimension of latent (z) space)
                * dataset (DatasetFeed object for training)
            * d_hyperparameters (optional dictionary of updates to `DEFAULTS`)
            * model_to_restore (filename of model to generate from (provide filename, without .meta)
        """
        self.sess = tf.Session()
        self.__dict__.update(Autoencoder.DEFAULTS, **d_hyperparams)
        self.analysis_folder = analysis_dir
        self.training_log = TrainingLog(self.analysis_folder)

        if build_dict:
            assert not model_to_restore
            assert all(key in build_dict for key in ['encoder', 'decoder', 'n_input', 'input_placeholder',
                                                     'shifted_input_placeholder', 'dataset', 'model_folder'])
            self.encoder = build_dict['encoder']
            self.decoder = build_dict['decoder']
            self.input_placeholder = build_dict['input_placeholder']
            self.shifted_input_placeholder = build_dict['shifted_input_placeholder']

            self.n_input = build_dict['n_input']
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
                tf.add_to_collection(Autoencoder.RESTORE_KEY, handle)
            self.sess.run(tf.initialize_all_variables())

            # unpack handles for tensor ops to feed or fetch
            (_1, _2,  # input_placeholder, shifted_input_placeholder
             self.encoding_cell, self.encoding_hidden, self.x_reconstructed, self.encoding_cell_, self.encoding_hidden_, self.x_reconstructed_,
             self.cost, self.rec_loss, self.l2_reg, self.apply_gradients_op, _3) = handles  # Last one is global_step
        elif model_to_restore:
            assert not build_dict
            self.model_folder = '/'.join((model_to_restore.split('/')[:-1])) + '/'
            with open(self.model_folder + '/network_settings.json') as network_json_file:
                json_vector_settings_dict = json.load(network_json_file)
            model_datetime = json_vector_settings_dict['model_datetime']
            self.datetime = "{}_reloaded".format(model_datetime)

            # Load the cost history
            self.training_log.load_costs_from_file()

            # rebuild graph
            meta_graph = os.path.abspath(model_to_restore)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.sess, meta_graph)

            handles = self.sess.graph.get_collection(Autoencoder.RESTORE_KEY)
            print("Restored handles: ", handles)
            (self.input_placeholder, self.shifted_input_placeholder,
             self.encoding_cell, self.encoding_hidden, self.x_reconstructed, self.encoding_cell_, self.encoding_hidden_,
             self.x_reconstructed_, self.cost, self.rec_loss, self.l2_reg, self.apply_gradients_op, self.global_step) = handles

            self.optimizer, self.gradient_acc, apply_gradients_op = \
                setup_AE_training_ops(self.learning_rate, self.cost, self.global_step)
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
            n_steps = self.input_placeholder.get_shape()[0].value  # To convert to int (otherwise this returns Dimension object)

            print("input, shifted_input shape: ", self.input_placeholder.get_shape(),
                  self.shifted_input_placeholder.get_shape())

            encoding = self.encoder(self.input_placeholder)  # LSTMStateTuple(c=(batch_size, n_hidden), h=(batch_size, n_hidden))

            print('encoding: ', encoding)

            print("Finished setting up encoder")
            print([var._variable for var in tf.all_variables()])

            # decoding
            # reconstruction is (n_steps, batch_size, n_outputs)
            reconstruction = self.decoder(encoding, inputs=self.shifted_input_placeholder)  # Feed the ground truth to the decoder
            # reconstruction = self.decoder(z_mean, inputs=x_shifted)  # When KL cost is eliminated, can just use the mean

            # Reconstruction is (n_steps, batch_size, n_inputs)

            print("Finished setting up decoder")
            print([var._variable for var in tf.all_variables()])

            # reconstruction loss: mismatch b/w x & reconstruction
            # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis

            print('reconstruction shape: ', reconstruction.get_shape())  # (n_steps, batch_size, n_outputs)
            print('input shape: ', self.input_placeholder.get_shape())  # (n_steps, batch_size, n_outputs)

            # rec_loss = VAE.crossEntropy(reconstruction, x_in)
            rec_loss = tf.reduce_mean((reconstruction - self.input_placeholder)**2, name="vae_cost")

            with tf.name_scope("l2_regularization"):
                regularizers = [tf.nn.l2_loss(var) for var in self.sess.graph.get_collection(
                    "trainable_variables") if "weights" in var.name]
                l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

            with tf.name_scope("cost"):
                cost = rec_loss + l2_reg

            print("Defined loss functions")

            # optimization
            self.optimizer, self.gradient_acc, apply_gradients_op = setup_AE_training_ops(self.learning_rate, cost, self.global_step)
            print("Defined training ops")

            print([var._variable for var in tf.all_variables()])

            # ops to directly explore encoding space
            with tf.name_scope("encoding_in"):
                encoding_ = rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, shape=[None, self.decoder.n_LSTM_hidden], name="cell_encoding_in"),
                                                    tf.placeholder(tf.float32, shape=[None, self.decoder.n_LSTM_hidden], name='hidden_encoding_in'))
            graph_scope.reuse_variables()  # No new variables should be created from this point on
            x_reconstructed_ = self.decoder(encoding_)

            return (self.input_placeholder, self.shifted_input_placeholder, encoding[0], encoding[1], reconstruction,  # Removed dropout from second place
                    encoding_[0], encoding_[1], x_reconstructed_, cost, rec_loss, l2_reg, apply_gradients_op, self.global_step)

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

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.input_placeholder: x}
        return self.sess.run(rnn_cell.LSTMStateTuple(self.encoding_cell, self.encoding_hidden), feed_dict=feed_dict)

    def decode(self, encoding):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        feed_dict = {self.encoding_cell_: encoding[0], self.encoding_hidden_: encoding[1]}
        return self.sess.run(self.x_reconstructed_, feed_dict=feed_dict)

    def ae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(self.encode(x))

    def train(self, max_iter=np.inf, max_epochs=np.inf, verbose=True, save=True):

        # Get ops for gradient updates
        update_gradients_ops = self.gradient_acc.update_gradients_ops()
        clear_gradients_ops = self.gradient_acc.clear_gradients()

        if save:
            self.saver = tf.train.Saver(tf.all_variables())

        outdir = self.model_folder
        self.accumulated_cost = 0
        now = datetime.now().isoformat()[11:]
        print("------- Training begin: {} -------\n".format(now))

        self.batch_count = 0  # For counting when to apply gradients
        while True:
            try:
                x = self.dataset.next_batch()  # (batch_size, n_steps, n_inputs)
                x = np.transpose(x, [1, 0, 2])  # (n_steps, batch_size, n_inputs)

                total_batch_size = x.shape[1]
                n_inputs = x.shape[2]
                x_shifted = np.concatenate((np.zeros((1, total_batch_size, n_inputs)), x[0:-1, :, :]), axis=0)

                # Reverse the input to the encoder in time!
                x = x[::-1, :, :]
                # Quick tests
                if self.batch_size > 6:
                    assert x[10, 6, 10] == x_shifted[-10, 6, 10]
                    assert x[4, 3, 2] == x_shifted[-4, 3, 2]
                assert x.shape == x_shifted.shape

                if self.max_batch_size_for_gradients and total_batch_size > self.max_batch_size_for_gradients:
                    pass

                feed_dict = {self.input_placeholder: x, self.shifted_input_placeholder: x_shifted}

                print("Updating gradients...")

                fetches = [self.x_reconstructed, self.cost, self.rec_loss, self.global_step] + update_gradients_ops
                x_reconstructed, cost, rec_loss, i, *_ = self.sess.run(fetches, feed_dict=feed_dict)

                # print("Gradients before being cleared:")
                # print([(k, self.sess.run(self.gradient_acc._var_to_accum_grad[k], feed_dict=feed_dict))
                #        for k in self.gradient_acc._var_to_accum_grad])

                if self.batch_count >= self.num_batches_per_grad_update:
                    print("Applying gradients...", end='')
                    self.sess.run(self.apply_gradients_op, feed_dict=feed_dict)
                    print("done")
                    self.sess.run(clear_gradients_ops)
                    self.batch_count = 0

                    print("Step {}-> cost for this minibatch: {}".format(i, cost))

                # print("Gradients should be zero: ")
                # print([(k, self.sess.run(self.gradient_acc._var_to_accum_grad[k], feed_dict=feed_dict))
                #        for k in self.gradient_acc._var_to_accum_grad])

                # if i % 10 == 0 and verbose:
                #     print("Step {}-> cost for this minibatch: {}".format(i, cost))
                #     print("   minibatch KL_cost = {}, reconst = {}".format(np.mean(kl_loss),
                #                                                            np.mean(rec_loss)))

                self.training_log.update_costs({'total_cost_history': cost, 'reconstruction_cost_history': rec_loss})


                self.batch_count += 1

                if i % 250 == 0:
                    # SAVE ALL MODEL PARAMETERS
                    self.save_model(outdir)

                if i % 5 == 0:
                    # SAVE COSTS FOR LEARNING CURVES
                    np.save(self.analysis_folder + 'total_cost.npy', self.training_log.total_cost_history)
                    np.save(self.analysis_folder + 'reconstruction_cost.npy', self.training_log.reconstruction_cost_history)

                if i >= max_iter or self.dataset.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, self.dataset.epochs_completed, self.accumulated_cost / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    self.save_model(outdir)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError):  # not logging
                        pass
                    return cost

            except(KeyboardInterrupt):
                print("\nTraining temporarily paused, enter an option:")
                print("1 - Terminate training")
                print("2 - Change learning rate (current is {})".format(self.learning_rate))
                option = None
                while not option in ['1', '2']:
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
