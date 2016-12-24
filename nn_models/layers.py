import tensorflow as tf
from utils.functionaltools import composeAll


class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)

    @staticmethod
    def wbVars(fan_in: int, fan_out: int):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))


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

