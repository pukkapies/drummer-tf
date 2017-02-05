import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell import RNNCell

tf.report_uninitialized_variables()

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
        self.cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=lstm_activation)
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
        assert len(inputs.get_shape()) == 3
        # n_steps = input.get_shape()[0]
        # n_input = input.get_shape()[2]
        # print('n_steps', n_steps)

        # print('input shape:', input.get_shape())

        # Prepare data shape to match `rnn` function requirements
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # x = tf.unpack(inputs)
        # print('after splitting, x length and shape: ', [len(x), x[0].get_shape()])
        # print('state: ', init_state)

        # Get lstm cell output
        # NB outputs is a list of length n_steps of network outputs, state is just the final state
        with tf.variable_scope(self.scope, initializer=self.initializer) as scope:
            try:
                outputs, final_state = rnn.dynamic_rnn(self.cell, inputs, initial_state=init_state,
                                                       dtype=tf.float32, time_major=True)
            except ValueError:  # RNN was already initialised, so share variables
                scope.reuse_variables()
                outputs, final_state = rnn.dynamic_rnn(self.cell, inputs, initial_state=init_state,
                                                       dtype=tf.float32, time_major=True)
        # print('outputs shape:', outputs[0].get_shape())
        # print('states shape:', final_state[0].get_shape()) # (batch_size, n_hidden) NB This has [0] because it is LSTMStateTuple

        # final_output = tf.pack(outputs)
        # print(final_output.get_shape())  # (num_steps, batch_size, n_hidden)

        return outputs, final_state

##################################################################################################################

class Stacked_LSTM(SimpleLSTM):
    """
    Uses the Tensorflow function rnn_cell.MultiRNNCell to create a stacked LSTM architecture
    """
    pass


##################################################################################################################


class LSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        """
        Calls the LSTM, computing outputs for given inputs and initial state
        :param inputs: Tensor of shape (batch_size, n_inputs)
        :param state: Tuple of 2 tensors of shape (batch_size, n_hidden). Order is (cell_state, hidden_state)
        :param scope: name scope
        :return: Output, new (cell, hidden) state tuple
        """
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            input_size = inputs.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [input_size, 4 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat(1, [inputs, h])
            W_both = tf.concat(0, [W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias

            i, j, f, o = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)

class BNLSTMCell(RNNCell):
    """Batch normalized LSTM as described in arxiv.org/abs/1603.09025"""
    def __init__(self, num_units, training_flag):
        self.num_units = num_units
        self.training_flag = training_flag

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        """
        Calls the LSTM, computing outputs for given inputs and initial state
        :param inputs: Tensor of shape (batch_size, n_inputs)
        :param state: Tuple of 2 tensors of shape (batch_size, n_hidden). Order is (cell_state, hidden_state).
                        If None, then it sets the state to zeros
        :param scope: name scope
        :return: Output, new (cell, hidden) state tuple
        """
        if state == None:
            state = self.zero_state(inputs.get_shape()[0], tf.float32)

        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = inputs.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 4 * self.num_units],
                initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm(xh, 'xh', self.training_flag)
            bn_hh = batch_norm(hh, 'hh', self.training_flag)

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = batch_norm(new_c, 'c', self.training_flag)

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def bn_lstm_identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype)

    return _initializer

def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer

def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999):
    """Assume 2d [batch, values] tensor"""

    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[1]

        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])  # Should this be a constant zero initializer?

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)
