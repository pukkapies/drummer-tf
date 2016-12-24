import tensorflow as tf
from nn_models.layers import FeedForward, Dense
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
    def __init__(self, n_hidden, latent_size, initial_state=None):
        """
        Sets up an LSTM encode for the VAE
        :param n_hidden: Size of hidden layer of LSTM
        :param latent_size: Size of latent space
        """
        self.n_hidden = n_hidden
        self.latent_size = latent_size
        self.initial_state = initial_state

    def __call__(self, input):
        """
        Calls the LSTM encoder
        :param input: Tensor of shape (n_steps, batch_size, n_inputs)
        :param init_state: Initial state. Tuple of 2 tensors of shape (batch_size, n_hidden). Can be None,
                            in which case the initial state is set to zero
        :return: z_mean, z_log_sigma, both of size latent_size
        """
        # encoding / recognition model q(z|x)
        print("Before unpack, input shape: ", input.get_shape())
        x = tf.unpack(input)  # List of length n_steps, each element is (batch_size, n_inputs)
        n_steps = len(x)
        print("After unpack, input has length {} and elements shape".format(len(x)), x[0].get_shape())
        lstm_encoder = SimpleLSTM(self.n_hidden, scope='LSTM_encoder')
        # outputs (shape is (n_steps, batch_size, n_outputs)), final state
        outputs, final_state = lstm_encoder(input, self.initial_state)
        z_mean = Dense(scope="z_mean", size=self.latent_size, nonlinearity=tf.identity)(outputs[-1])
        z_log_sigma = Dense(scope="z_log_sigma", size=self.latent_size, nonlinearity=tf.identity)(outputs[-1])

        return z_mean, z_log_sigma