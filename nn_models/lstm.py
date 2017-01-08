import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from tensorflow.python.ops.math_ops import tanh


class SimpleLSTM(object):

    def __init__(self, n_hidden, scope='SimpleLSTM', lstm_activation=tanh, initializer=None):
        """
        Sets up the LSTM model with an additional output filter to shape to size n_outputs
        :param input_placeholder: Placeholder tensor of shape (n_steps, batch_size, n_inputs)
        :param state_placeholder: List (length num_layers) of a tuple of 2 placeholder tensors of shape (batch_size, n_hidden).
                Can be None, in which case, the LSTM is initialised with a zero state (see rnn.rnn implementation)
        :param n_hidden: size of the hidden layers of the LSTM
        :param lstm_activation: Activation function of the inner states of the LSTM
        """
        self.cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=lstm_activation)
        self.scope = scope
        self.lstm_activation = lstm_activation
        self.initializer = initializer

    def __call__(self, input, init_state):
        """
        Calls the RNN model, computing outputs for given inputs and initial state
        :param input: Tensor of shape (n_steps, batch_size, n_inputs)
        :param init_state: Initial state. Tuple of 2 tensors of shape (batch_size, n_hidden). Can be None,
                            in which case the initial state is set to zero
        :return: outputs (shape is (n_steps, batch_size, n_outputs)), final state
        """
        assert len(input.get_shape()) == 3
        # n_steps = input.get_shape()[0]
        # n_input = input.get_shape()[2]
        # print('n_steps', n_steps)

        # print('input shape:', input.get_shape())

        # Prepare data shape to match `rnn` function requirements
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        x = tf.unpack(input)
        # print('after splitting, x length and shape: ', [len(x), x[0].get_shape()])
        # print('state: ', init_state)

        # Get lstm cell output
        # NB outputs is a list of length n_steps of network outputs, state is just the final state
        with tf.variable_scope(self.scope, initializer=self.initializer) as scope:
            try:
                outputs, final_state = rnn.rnn(self.cell, x, initial_state=init_state, dtype=tf.float32)
            except ValueError:  # RNN was already initialised, so share variables
                scope.reuse_variables()
                outputs, final_state = rnn.rnn(self.cell, x, initial_state=init_state, dtype=tf.float32)
        # print('outputs shape:', outputs[0].get_shape())
        # print('states shape:', final_state[0].get_shape()) # (batch_size, n_hidden) NB This has [0] because it is LSTMStateTuple

        final_output = tf.pack(outputs)
        # print(final_output.get_shape())  # (num_steps, batch_size, n_hidden)

        return final_output, final_state

##################################################################################################################

class Stacked_LSTM(SimpleLSTM):
    """
    Uses the Tensorflow function rnn_cell.MultiRNNCell to create a stacked LSTM architecture
    """
    pass


##################################################################################################################

class ColahLSTM(object):
    # DEPRECATED
    def __init__(self, n_inputs, n_hidden, batch_size, scope):
        self.scope = scope
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        with tf.variable_scope('LSTM'):
            self.hidden_state = tf.zeros([self.batch_size, self.n_hidden])
            self.cell_state = tf.zeros([self.batch_size, self.n_hidden])
        self.variables = self._create_variables()

    def _create_variables(self):
        vars = dict()

        with tf.variable_scope(self.scope, reuse=None):
            vars['forget_weights'] = tf.get_variable('forget_weights', (self.n_inputs + self.n_hidden, self.n_hidden),
                                                     initializer=tf.random_normal_initializer())
            vars['forget_bias'] = tf.get_variable('forget_bias', (self.n_hidden, ),
                                                  initializer=tf.constant_initializer(1.0))

            vars['input_gate_weights'] = tf.get_variable('input_gate_weights',
                                                     (self.n_inputs + self.n_hidden, self.n_hidden),
                                                     initializer=tf.random_normal_initializer())
            vars['input_gate_bias'] = tf.get_variable('input_gate_bias', (self.n_hidden,),
                                                  initializer=tf.constant_initializer(1.0))

            vars['input_weights'] = tf.get_variable('input_weights',
                                                     (self.n_inputs + self.n_hidden, self.n_hidden),
                                                     initializer=tf.random_normal_initializer())
            vars['input_bias'] = tf.get_variable('input_bias', (self.n_hidden,),
                                                  initializer=tf.constant_initializer(0.0))

            vars['output_weights'] = tf.get_variable('output_weights',
                                                     (self.n_inputs + self.n_hidden, self.n_hidden),
                                                     initializer=tf.random_normal_initializer())
            vars['output_bias'] = tf.get_variable('output_bias', (self.n_hidden,),
                                                  initializer=tf.constant_initializer(0.0))
        return vars

    def update(self, input_vec):
        """
        Performs the update rule for the LSTM
        :param input_vec: Tensor of shape (batch_size, n_inputs)
        :return: Output - Tensor of shape (batch_size, n_hidden)
        """
        with tf.variable_scope(self.scope, reuse=True):
            # print('input_vec shape:', input_vec.get_shape())
            # input_vec_split = tf.split(0, self.batch_size, input_vec)
            # print('after splitting: ', input_vec_split)
            # print('first input_vec_split shape: ', tf.squeeze(input_vec_split[0]).get_shape())
            # print('hiddenstate check: ', self.hidden_state.get_shape())
            # concat_hidden_input = tf.pack([tf.concat(0, [tf.squeeze(vec), self.hidden_state])
            #                                for vec in input_vec_split])

            concat_hidden_input = tf.concat(1, [input_vec, self.hidden_state])

            forget_activation = tf.sigmoid(tf.matmul(concat_hidden_input, self.variables['forget_weights'])
                                           + self.variables['forget_bias'])

            self.cell_state = self.cell_state * forget_activation

            input_activation = tf.sigmoid(tf.matmul(concat_hidden_input, self.variables['input_gate_weights'])
                                          + self.variables['input_gate_bias']) * \
                               tf.tanh(tf.matmul(concat_hidden_input, self.variables['input_weights'])
                                       + self.variables['input_bias'])
            self.cell_state = self.cell_state + input_activation

            output_activation = tf.sigmoid(tf.matmul(concat_hidden_input, self.variables['output_weights'])
                                           + self.variables['output_bias'])
            output = tf.tanh(self.cell_state) * output_activation

            self.hidden_state = output
        return output
