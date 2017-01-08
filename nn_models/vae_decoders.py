import tensorflow as tf
from nn_models.layers import FeedForward, Dense
from nn_models.lstm import SimpleLSTM
from nn_models.initialisers import wbVars_Xavier
from tensorflow.python.ops import variable_scope as vs
from utils.functionaltools import composeAll


class FeedForwardDecoder(object):

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

    def __call__(self, z):
        decoding = FeedForward(scope="decoding", sizes=self.architecture[1:-1], dropout=self.dropout,
                               nonlinearity=self.nonlinearity)
        h_decoded = decoding(z)

        x_reconstructed = tf.identity(Dense("x_decoding", self.architecture[0], self.dropout, self.squashing)(h_decoded),
                                      name='x_reconstructed')
        return x_reconstructed


class LSTMDecoder(object):
    def __init__(self, n_LSTM_hidden, postlatent_dense_layers, n_outputs, n_steps, output_activation=tf.tanh):
        """
        Sets up an LSTM encode for the VAE
        :param n_LSTM_hidden: Size of hidden layer of LSTM
        :param n_outputs: Size of inputs/outputs for each time step
        :param n_steps: int, number of steps to run the LSTM decoder
        :param output_activation: Activation function to apply to final LSTM output
        """
        self.n_LSTM_hidden = n_LSTM_hidden
        self.postlatent_dense_layers = postlatent_dense_layers
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.output_activation = output_activation

    def __call__(self, z):
        """
        Calls the LSTM decoder
        :param z: Tensor of shape (batch_size, latent_size)
        :return: final_outputs: Tensor  of shape (n_steps, batch_size, n_hidden)
        """
        # encoding / recognition model q(z|x)
        batch_size = z.get_shape()[0]

        postlatent_layers = [Dense(scope="postlatent_{}".format(i), size=hidden_size, nonlinearity=tf.tanh,
                                  initialiser=wbVars_Xavier) for i, hidden_size in enumerate(reversed(self.postlatent_dense_layers))]
        initial_LSTM_states_encoded = composeAll(postlatent_layers)(z)

        # with tf.variable_scope('LSTM_decoder') as decoder_scope:
        lstm_decoder = SimpleLSTM(self.n_LSTM_hidden, scope="LSTM_decoder")
        lstm_activation = lstm_decoder.lstm_activation

        # (Cell state, hidden state):
        init_states = (Dense(scope="latent_to_LSTM_cell", size=self.n_LSTM_hidden, nonlinearity=lstm_activation,
                             initialiser=wbVars_Xavier)(initial_LSTM_states_encoded),
                       Dense(scope="latent_to_LSTM_hidden", size=self.n_LSTM_hidden, nonlinearity=lstm_activation,
                             initialiser=wbVars_Xavier)(initial_LSTM_states_encoded))

        # input = tf.placeholder(tf.float32, shape=[1, batch_size, self.n_outputs])  # (n_steps, batch_size, n_inputs)
        # states = (tf.placeholder(tf.float32, shape=[batch_size, self.n_hidden]),
        #           tf.placeholder(tf.float32, shape=[batch_size, self.n_hidden]))

        first_input = tf.zeros((1, batch_size, self.n_outputs))  # NB Just one step, so first argument is 1
        dense_output = Dense(scope="LSTM_decoder/dense_output", size=self.n_outputs, nonlinearity=self.output_activation,
                             initialiser=wbVars_Xavier)

        # feed_dict = {input: first_input, states[0]: first_states[0], states[1]: first_states[1]}
        lstm_input = first_input
        states = init_states
        final_outputs = []
        for step in range(self.n_steps):
            # print("STEP: ", step)
            outputs, final_state = lstm_decoder(lstm_input, states)
            outputs_list = tf.unpack(outputs)  # List of length 1, element shape (batch_size, n_hidden)
            final_output = dense_output(outputs_list[0])  # (batch_size, n_outputs)
            final_outputs.append(final_output)
            lstm_input = tf.pack([final_output])  # (1, batch_size, n_outputs)
            states = final_state

        final_outputs = tf.pack(final_outputs)  # (n_steps, batch_size, n_outputs)
        print('final outputs shape from decoder: ', final_outputs.get_shape())
        return final_outputs
