import tensorflow as tf
from nn_models.layers import FeedForward, Dense
from nn_models.lstm import SimpleLSTM
from nn_models.initialisers import wbVars_Xavier
from utils.functionaltools import composeAll


class AE_FeedForwardEncoder(object):

    def __init__(self, architecture, nonlinearity):
        """
        Sets up an MLP encoder for the VAE
        :param architecture: (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D encoding,
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
        final_encoding = Dense("encoding", self.architecture[-1], self.dropout)(h_encoded)
        return final_encoding


class AE_LSTMEncoder(object):

    def __init__(self, n_LSTM_hidden, dense_layers=[], initial_state=None):
        """
        Sets up an LSTM encode for the VAE
        :param n_LSTM_hidden: Size of hidden layer of LSTM
        :param dense_layers: List of hidden layer sizes of dense layers feeding into latent mean/stdev.
                            Last element is encoding dimension. If empty then the last LSTM state is the encoding
        """
        assert type(dense_layers) == list
        if len(dense_layers) != 0:
            raise NotImplementedError("Dense layers not yet implemented for vanilla autoencoder.")
        self.n_LSTM_hidden = n_LSTM_hidden
        self.dense_layers = dense_layers
        self.initial_state = initial_state

    def __call__(self, input):
        """
        Calls the LSTM encoder
        :param input: Tensor of shape (n_steps, batch_size, n_inputs)
        :param init_state: Initial state. Tuple of 2 tensors of shape (batch_size, n_hidden). Can be None,
                            in which case the initial state is set to zero
        :return: final_state tuple of the LSTM
        """
        # encoding model
        print("Before unpack, input shape: ", input.get_shape())
        # print("After unpack, input has length {} and elements shape".format(len(x)), x[0].get_shape())
        lstm_encoder = SimpleLSTM(self.n_LSTM_hidden, scope='LSTM_encoder',
                                  initializer=tf.contrib.layers.xavier_initializer())
        # outputs (shape is (n_steps, batch_size, n_outputs)), final state
        outputs, final_state = lstm_encoder(input, self.initial_state)
        # outputs = tf.unpack(outputs)  # List of length n_steps

        # Dense layers to feed into latent variable mean and standard deviation
        # layers = [Dense(scope="prelatent_{}".format(i), size=hidden_size, nonlinearity=tf.tanh,
        #                     initialiser=wbVars_Xavier) for i, hidden_size in enumerate(reversed(self.dense_layers))]
        # z_mean_log_sigma_encoded = composeAll(layers)(outputs[-1])

        return final_state