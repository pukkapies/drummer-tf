import tensorflow as tf
from nn_models.layers import FeedForward, Dense


class FeedForwardEncoder(object):

    def __init__(self, architecture, nonlinearity):
        """
        Sets up an MLP encoder for the VAE
        :param architecture: (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
               & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]
        :param nonlinearity: Nonlinear activation function to use in the MLP
        """
        self.architecture = architecture
        self.nonlinearity = nonlinearity
        self.dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

    def __call__(self, x_in):
        # encoding / "recognition": q(z|x)
        assert x_in.get_shape().value == self.architecture[0]

        encoding = FeedForward(scope="dense_layer", sizes=self.architecture[1:-1], dropout=self.dropout,
                               nonlinearity=self.nonlinearity)
        h_encoded = encoding(x_in)
        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = Dense("z_mean", self.architecture[-1], self.dropout)(h_encoded)
        z_log_sigma = Dense("z_log_sigma", self.architecture[-1], self.dropout)(h_encoded)
        return z_mean, z_log_sigma


class LSTMEncoder(object):

    def __init__(self):