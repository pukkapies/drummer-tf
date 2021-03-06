import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell


class SimpleLSTM(object):

    def __init__(self, n_hidden, scope='SimpleLSTM', lstm_activation=tanh, initializer=None):
        """
        Sets up the LSTM model with an additional output filter to shape to size n_outputs
        :param input_placeholder: Placeholder tensor of shape (n_steps, batch_size, n_inputs)
        :param state_placeholder: List (length num_layers) of a tuple of 2 placeholder tensors of shape (batch_size, n_hidden).
                Can be None, in which case, the LSTM is initialised with a zero state (see rnn.rnn implementation)
        :param n_hidden: size of the hidden layers of the LSTM
        :param lstm_activation: Activation function of the inner states of the LSTM
                (determines the range of values stored in the hidden states)
        """
        self.cell = BasicLSTMCell(n_hidden, forget_bias=1.0, activation=lstm_activation)
        self.scope = scope
        self.lstm_activation = lstm_activation
        self.initializer = initializer

    def __call__(self, inputs, init_state):
        """
        Calls the RNN model, computing outputs for given inputs and initial state
        :param inputs: Tensor of shape (n_steps, batch_size, n_inputs)
        :param init_state: Initial state. Tuple of 2 tensors of shape (batch_size, n_hidden). Can be None,
                            in which case the initial state is set to zero. Order is (cell_state, hidden_state)
        :return: outputs (shape is (n_steps, batch_size, n_outputs)), final state
        """
        with tf.variable_scope(self.scope, initializer=self.initializer) as scope:
            outputs, final_state = rnn.dynamic_rnn(self.cell, inputs, initial_state=init_state,
                                                   dtype=tf.float32, time_major=True)
        return outputs, final_state


#################################################################################################################


class LSTMDecoder(object):
    def __init__(self, n_LSTM_hidden, n_outputs, n_steps=None, output_activation=tf.tanh):
        """
        Sets up an LSTM encode for the VAE
        :param n_LSTM_hidden: Size of hidden layer of LSTM
        :param n_outputs: Size of inputs/outputs for each time step
        :param n_steps: int, number of steps to run the LSTM decoder. Not needed if inputs are provided when called.
        :param output_activation: Activation function to apply to final LSTM output
        """
        self.n_LSTM_hidden = n_LSTM_hidden
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.output_activation = output_activation

    def __call__(self, z, inputs=None):
        """
        Calls the LSTM decoder
        :param z: Tensor of shape (batch_size, latent_size)
        :param inputs: Optional inputs to feed to the LSTM of shape (n_steps, batch_size, n_inputs).
                        batch_size must match z batch_size, n_inputs must match n_LSTM_hidden.
                        If not present, the LSTM will feed in its own output from the previous step.
        :return: final_outputs: Tensor  of shape (n_steps, batch_size, n_hidden)
        """
        # encoding / recognition model q(z|x)
        batch_size = z.get_shape()[0]

        if inputs is not None:
            inputs_batch_size = inputs.get_shape()[1]
            inputs_n_inputs = inputs.get_shape()[2]
            assert batch_size == inputs_batch_size
            assert self.n_outputs == inputs_n_inputs, "LSTM has been set up with different output size to the " \
                                                      "attempted size of input"
        else:
            assert self.n_steps, "LSTMDecoder called without inputs, but n_steps has not been set."

        with tf.variable_scope("LSTM_Decoder") as decoder_scope:
            lstm_decoder = SimpleLSTM(self.n_LSTM_hidden, initializer=tf.contrib.layers.xavier_initializer())
            lstm_activation = lstm_decoder.lstm_activation  # Determines the range of the LSTM hidden state

            # (Cell state, hidden state):
            init_states = (Dense(scope="latent_to_LSTM_cell", size=self.n_LSTM_hidden, nonlinearity=lstm_activation)
                           (z),
                           Dense(scope="latent_to_LSTM_hidden", size=self.n_LSTM_hidden, nonlinearity=lstm_activation)
                           (z))

            dense_output = Dense(scope="dense_output", size=self.n_outputs, nonlinearity=self.output_activation,
                                 initialiser=wbVars_Xavier)

            states = init_states

            if inputs is not None:
                outputs, final_state = lstm_decoder(inputs, states)
                final_outputs = dense_output(outputs)
            else:
                first_input = tf.zeros((1, batch_size, self.n_outputs))  # NB Just one step, so first argument is 1

                lstm_input = first_input
                final_outputs = []
                for step in range(self.n_steps):
                    outputs, final_state = lstm_decoder(lstm_input, states)
                    outputs_list = tf.unpack(outputs)  # List of length 1, element shape (batch_size, n_hidden)
                    final_output = dense_output(outputs_list[0])  # (batch_size, n_outputs)
                    final_outputs.append(final_output)
                    lstm_input = tf.pack([final_output])  # (1, batch_size, n_outputs)
                    states = final_state
                    decoder_scope.reuse_variables()

                final_outputs = tf.pack(final_outputs)  # (n_steps, batch_size, n_outputs)

            print('final outputs shape from decoder: ', final_outputs.get_shape())
            return final_outputs


#################################################################################################################


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    RESTORE_KEY = "to_restore"

    def __init__(self, build_dict=None, scope='VAE', model_to_restore=False):
        """(Re)build a symmetric VAE model with given:
            * build_dict (if the model is being built new. The dict should contain the following keys:
                * encoder (callable object that takes input tensor as argument and returns tensors z_mean, z_log_sigma
                * decoder (callable object that takes z as input and returns reconstructed x)
                * input_size (number of inputs at each time step)
                * input_placeholder (placeholder object for inputs)
                * latent_size (dimension of latent (z) space)
                * dataset (object for training)
            * model_to_restore (filename of model to generate from (provide filename, without .meta)
        """
        self.sess = tf.Session()

        if build_dict:
            assert not model_to_restore
            assert 'dataset' in build_dict
            assert all(key in build_dict for key in ['z_placeholder', 'decoder', 'n_input',
                                                     'input_placeholder', 'latent_dim',
                                                     'dataset', 'model_folder'])
            self.z_ph = build_dict['z_placeholder']
            self.decoder = build_dict['decoder']
            self.input_placeholder = build_dict['input_placeholder']
            self.shifted_input_placeholder = build_dict['shifted_input_placeholder']

            self.n_input = build_dict['n_input']
            self.latent_dim = build_dict['latent_dim']
            self.dataset = build_dict['dataset']
            self.batch_size = self.dataset.minibatch_size
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            # build graph
            self.scope = scope
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sess.run(tf.initialize_all_variables())

            # unpack handles for tensor ops to feed or fetch
            (_1, _2, _3,
             self.x_reconstructed, self.z_, self.x_reconstructed_,
             self.cost, self.apply_gradients_op, _4) = handles
        elif model_to_restore:
            assert not build_dict
            # rebuild graph
            tf.train.import_meta_graph(model_to_restore + ".meta").restore(self.sess, model_to_restore)

            handles = self.sess.graph.get_collection(VAE.RESTORE_KEY)
            print("Restored handles: ", handles)
            (self.input_placeholder, self.shifted_input_placeholder, self.z_ph,
             self.x_reconstructed, self.z_, self.x_reconstructed_,
             self.cost, self.apply_gradients_op, self.global_step) = handles
        else:
            raise Exception("VAE must be initialised with either build_dict or model_to_restore")

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sess)

    def _buildGraph(self):
        with tf.variable_scope(self.scope) as graph_scope:
            # Both will have size (n_steps, batch_size, n_inputs)
            n_steps = self.input_placeholder.get_shape()[0].value  # To convert to int (otherwise this returns Dimension object)

            # decoding / "generative": p(x|z)
            # reconstruction is (n_steps, batch_size, n_outputs)
            reconstruction = self.decoder(self.z_ph, inputs=self.shifted_input_placeholder)  # Feed the ground truth to the decoder
            print("Finished setting up decoder")

            cost = tf.reduce_mean((reconstruction - self.input_placeholder)**2, reduction_indices=[0, 1, 3])
            print("Defined loss functions")

            # optimization
            self.optimizer, apply_gradients_op = setup_VAE_training_ops(self.sess, self.learning_rate, cost, self.global_step)
            print("Defined training ops")

            # ops to directly explore latent space
            # defaults to prior z ~ N(0, I)
            with tf.name_scope("latent_in"):
                z_ = tf.placeholder_with_default(tf.random_normal([1, self.latent_dim]), # batch_size = 1 !
                                                 shape=[1, self.latent_dim])
            graph_scope.reuse_variables()  # No new variables to be created from this point on
            x_reconstructed_ = self.decoder(z_)

            return (self.input_placeholder, self.shifted_input_placeholder, self.z_ph,
                    reconstruction, z_, x_reconstructed_, cost, apply_gradients_op, self.global_step)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        feed_dict = dict()
        if zs is not None:
            feed_dict.update({self.z_: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sess.run(self.x_reconstructed_, feed_dict=feed_dict)

    def train(self, max_iter=np.inf, max_epochs=np.inf):
        while True:
            try:
                x = self.dataset.next_batch()  # (batch_size, n_steps, n_inputs)
                x = np.transpose(x, [1, 0, 2])  # (n_steps, samples_per_batch * batch_size, n_inputs)

                total_batch_size = x.shape[1]  # batch_size
                n_inputs = x.shape[2]
                x_shifted = np.concatenate((np.zeros((1, total_batch_size, n_inputs)), x[0:-1, :, :]), axis=0)

                feed_dict = {self.input_placeholder: x, self.shifted_input_placeholder: x_shifted}

                print("Updating gradients...")
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.apply_gradients_op]

                x_reconstructed, cost, i, _ = self.sess.run(fetches, feed_dict=feed_dict)

                print("Step {}-> cost for this minibatch: {}".format(i, cost))

                if i >= max_iter or self.dataset.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, self.dataset.epochs_completed, self.cost))
                    self.save_model(outdir)
                    return cost
            except(KeyboardInterrupt):
                print("final cost (@ step {} = epoch {}): {}".format(
                    i, self.dataset.epochs_completed, self.cost))
                self.save_model(self.outdir)
                print("------- Training end: {} -------\n".format(now))
                return cost


#################################################################################################################

################ THIS IS WHERE THE MODEL IS RUN, e.g. VIA COMMAND LINE ARGUMENTS ################



batch_size = args.batch_size # Comes from command line
dataset = .... # Comes from somewhere

n_steps = dataset.data_shape[0]
n_input = n_outputs = dataset.data_shape[1] # vector size

if args.create_new_model:  # Create a new model
    latent_dim = args.latent_dimension
    n_hidden_decoder = args.lstm_decoder_hidden_units

    decoder = LSTMDecoder(n_hidden_decoder, n_outputs, n_steps=n_steps, output_activation=tf.sigmoid)

    z_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, latent_dim])

    input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, batch_size, n_input], name="input")
    # The following placeholder is for feeding the ground truth to the LSTM decoder - the first input should be zeros
    shifted_input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, batch_size, n_input], name="shifted_input")

    build_dict = {'z_placeholder': z_ph,
                  'decoder': decoder,
                  'n_input': n_input,
                  'input_placeholder': input_placeholder,
                  'shifted_input_placeholder': shifted_input_placeholder,
                  'latent_dim': latent_dim,
                  'dataset': dataset}

    vae = VAE(build_dict=build_dict)
    cost = vae.train(max_iter=args.num_training_steps)
elif args.resume_model:  # Resume a previously trained model
    vae = VAE(model_to_restore=meta_graph)
    vae.dataset = dataset
    cost = vae.train(max_iter=args.num_training_steps)
else:
    # Load a previously trained model and sample from it
    vae = VAE(model_to_restore=meta_graph)
    z_input = ... # whatever
    sample = vae.decode(zs=z_input)
