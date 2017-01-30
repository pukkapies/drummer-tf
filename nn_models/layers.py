import tensorflow as tf
from utils.functionaltools import composeAll


class Dense():
    """Fully-connected layer. Can be applied to Tensors of shape ("""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity, initialiser=None):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        assert initialiser, "Must specify an initialiser for Dense layer"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity
        self.initialiser = initialiser

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.variable_scope(self.scope):
            self.w, self.b = self.initialiser(x.get_shape()[-1].value, self.size)
            # self.w = tf.nn.dropout(self.w, self.dropout)
            try:
                return self.nonlinearity(tf.matmul(x, self.w) + self.b)
            except ValueError:  # This deals with the case where x has shape (n_steps, batch_size, n_inputs)
                return tf.map_fn(lambda _: self.nonlinearity(tf.matmul(_, self.w) + self.b), x)


class FeedForward():
    """Feedforward network/MLP"""
    def __init__(self, scope="dense_layer", sizes=None, dropout=1., nonlinearity=None):
        """
        Initializer
        :param scope: string, or list of strings. If a list, then must be the same length as sizes, and each
                        string entry will be used as namescope for each hidden layer
        :param sizes: list of layer sizes. First entry is first hidden layer size, last layer is output.
                        The input size is obtained when the class is called, so is not needed in the initialisation.
        :param dropout: Keep rate
        :param nonlinearity: Nonlinear activation function to be used in the MLP
        """
        assert sizes, "Need to specify layer sizes for Feedforward architecture"
        assert nonlinearity, "Need to specify nonlinearity for Feedforward architecture"
        self.sizes = sizes[::-1]  # Reverse the order for function composition
        self.dropout = dropout
        self.nonlinearity = nonlinearity
        assert type(scope)==str or type(scope)==list
        if type(scope)==list:
            assert len(scope)==len(sizes)
            self.scope = scope
        else:
            self.scope = [scope] * len(sizes)

    def __call__(self, input):
        # tf.Tensor -> tf.Tensor
        layer_fns = [Dense(self.scope[i], hidden_size, self.dropout, self.nonlinearity)
                    # hidden layers reversed for function composition: outer -> inner
                    for i, hidden_size in enumerate(self.sizes)]
        output = composeAll(layer_fns)(input)
        return output

