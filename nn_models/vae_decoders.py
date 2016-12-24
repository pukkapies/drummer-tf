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
