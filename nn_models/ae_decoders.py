import tensorflow as tf
from nn_models.layers import FeedForward, Dense
from nn_models.lstm import SimpleLSTM
from nn_models.initialisers import wbVars_Xavier
from tensorflow.python.ops import variable_scope as vs
from utils.functionaltools import composeAll


class AE_FeedForwardDecoder(object):

    def __init__(self, architecture, nonlinearity, squashing):
        """
        Sets up an MLP decoder for the VAE
        :param architecture: architecture: (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
               & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]
        :param nonlinearity: Nonlinear activation function inside MLP
        :param squashing: Nonlinear activation function to apply to output
        """
        self.architecture = architecture
        self.nonlinearity = nonlinearity  # Inner nonlinear activations
        self.squashing = squashing  # Output squashing function
        self.dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

    def __call__(self, encoding):
        decoding = FeedForward(scope="decoding", sizes=self.architecture[1:-1], dropout=self.dropout,
                               nonlinearity=self.nonlinearity)
        h_decoded = decoding(encoding)

        x_reconstructed = tf.identity(Dense("x_decoding", self.architecture[0], self.dropout, self.squashing)(h_decoded),
                                      name='x_reconstructed')
        return x_reconstructed


class AE_LSTMDecoder(object):
    def __init__(self, n_LSTM_hidden, n_outputs, n_steps=None, dense_layers=[], output_activation=tf.tanh):
        """
        Sets up an LSTM encode for the VAE
        :param n_LSTM_hidden: Size of hidden layer of LSTM
        :param n_outputs: Size of inputs/outputs for each time step
        :param n_steps: int, number of steps to run the LSTM decoder. Not needed if inputs are provided when called.
        :param output_activation: Activation function to apply to final LSTM output
        """
        self.n_LSTM_hidden = n_LSTM_hidden
        self.dense_layers = dense_layers
        if len(self.dense_layers) != 0:
            raise NotImplementedError("Dense layers not yet implemented for AE LSTM decoder.")
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.output_activation = output_activation

    def __call__(self, encoding, inputs=None):
        """
        Calls the LSTM decoder
        :param encoding: Tuple of 2 tensors of shape (batch_size, n_hidden) for (cell_state, hidden_state)
        :param inputs: Optional inputs to feed to the LSTM of shape (n_steps, batch_size, n_inputs).
                        batch_size must match encoding batch_size, n_inputs must match n_LSTM_hidden.
                        If not present, the LSTM will feed in its own output from the previous step.
        :return: final_outputs: Tensor  of shape (n_steps, batch_size, n_hidden)
        """
        # encoding / recognition model q(z|x)
        c, h = encoding

        batch_size = c.get_shape()[0]
        assert h.get_shape()[0]==batch_size, "Batch size for hidden and cell states do not match"

        if inputs is not None:
            inputs_n_steps = inputs.get_shape()[0]
            inputs_batch_size = inputs.get_shape()[1]
            inputs_n_inputs = inputs.get_shape()[2]
            assert batch_size == inputs_batch_size
            assert self.n_outputs == inputs_n_inputs, "LSTM has been set up with different output size to the " \
                                                      "attempted size of input"
        else:
            assert self.n_steps, "LSTMDecoder called without inputs, but n_steps has not been set."

        with tf.variable_scope("LSTM_Decoder") as decoder_scope:
            # layers = [Dense(scope="postlatent_{}".format(i), size=hidden_size, nonlinearity=tf.tanh,
            #                 initialiser=wbVars_Xavier) for i, hidden_size in enumerate(reversed(self.dense_layers))]
            # initial_LSTM_states_encoded = composeAll(layers)(encoding)

            lstm_decoder = SimpleLSTM(self.n_LSTM_hidden, initializer=tf.contrib.layers.xavier_initializer())

            lstm_activation = lstm_decoder.lstm_activation  # Determines the range of the LSTM hidden state

            # (Cell state, hidden state):
            # init_states = (Dense(scope="latent_to_LSTM_cell", size=self.n_LSTM_hidden, nonlinearity=lstm_activation,
            #                      initialiser=wbVars_Xavier)(initial_LSTM_states_encoded),
            #                Dense(scope="latent_to_LSTM_hidden", size=self.n_LSTM_hidden, nonlinearity=lstm_activation,
            #                      initialiser=wbVars_Xavier)(initial_LSTM_states_encoded))

            # input = tf.placeholder(tf.float32, shape=[1, batch_size, self.n_outputs])  # (n_steps, batch_size, n_inputs)
            # states = (tf.placeholder(tf.float32, shape=[batch_size, self.n_hidden]),
            #           tf.placeholder(tf.float32, shape=[batch_size, self.n_hidden]))

            dense_output = Dense(scope="dense_output", size=self.n_outputs, nonlinearity=self.output_activation,
                                 initialiser=wbVars_Xavier)

            # feed_dict = {input: first_input, states[0]: first_states[0], states[1]: first_states[1]}

            states = (c, h)

            if inputs is not None:
                outputs, final_state = lstm_decoder(inputs, states)
                final_outputs = dense_output(outputs)  # outputs is (n_steps, batch_size, n_hidden)
            else:
                first_input = tf.zeros((1, batch_size, self.n_outputs))  # NB Just one step, so first argument is 1

                lstm_input = first_input
                final_outputs = []
                for step in range(self.n_steps):
                    # print("STEP: ", step)
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
