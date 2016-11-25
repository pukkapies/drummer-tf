import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


class SimpleLSTM(object):

    def __init__(self, input_placeholder, state_placeholder, n_hidden, n_outputs):
        """
        Sets up the LSTM model with an additional output filter to shape to size n_outputs
        :param input_placeholder: Placeholder tensor of shape (n_steps, batch_size, n_inputs)
        :param state_placeholder: List (length num_layers) of a tuple of 2 placeholder tensors of shape (batch_size, n_hidden).
                Can be None, in which case, the LSTM is initialised with a zero state (see rnn.rnn implementation)
        :param n_hidden: List of size of the hidden layers of the LSTM
        :param n_outputs: Size of the output
        """
        assert len(n_hidden) >= 1
        assert len(n_hidden) == len(state_placeholder)

        self.num_layers = len(n_hidden)
        self.weights_out = []
        self.biases_out = []
        self.cell = []
        print('n_outputs:', n_outputs)
        print('n_hidden:', n_hidden)

        # Turn n_outputs into a list for convenience
        if len(n_hidden)==1:
            n_outputs = [n_outputs]
        else:
            n_outputs = n_hidden[1:] + [n_outputs]

        for i in range(len(n_hidden)):
            with tf.variable_scope('LSTM_model/layer_{}'.format(i+1)):
            # with tf.variable_scope('LSTM_model'):
                self.weights_out.append(tf.Variable(tf.random_normal([n_hidden[i], n_outputs[i]])))
                self.biases_out.append(tf.Variable(tf.random_normal([n_outputs[i]])))
                # The following doesn't yet create variables, so doesn't use the variable_scope
                self.cell.append(rnn_cell.BasicLSTMCell(n_hidden[i], forget_bias=1.0))
                print(self.cell[i])

        self.prediction, self.state = self.sample(input_placeholder, state_placeholder)

    def sample(self, input, state):
        """
        Samples from the RNN model, computing outputs for given inputs and initial state
        :param input: Tensor of shape (n_steps, batch_size, n_inputs)
        :param state: Initial state. List (length num_layers) of a tuple of 2 tensors of shape (batch_size, n_hidden[i])
        :return: outputs (shape is (n_steps, batch_size, n_outputs)), states (list of final states for each layer)
        """
        n_steps = input.get_shape()[0]
        n_input = input.get_shape()[2]
        print('n_steps', n_steps)

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # print('input shape:', input.get_shape())
        # # Permuting batch_size and n_steps
        # x = tf.transpose(input, [1, 0, 2])
        print('input shape:', input.get_shape())
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(input, tf.pack([-1, n_input])) # The pack is necessary because this is a mixed list of int and Tensor
        print('x shape (post reshape):', x.get_shape())
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)
        print('after splitting, x length and shape: ', [len(x), x[0].get_shape()])
        print('state[0]: ', state[0])

        print('weights list length: ', len(self.weights_out))
        print('weights_out shape[0]:', self.weights_out[0].get_shape())

        final_states = []
        # Get lstm cell output
        for i in range(self.num_layers):
            with tf.variable_scope("LSTM_model/layer_{}".format(i+1)):
                # NB outputs is a list of length n_steps of network outputs, states is just the final state
                outputs, states = rnn.rnn(self.cell[i], x, initial_state=state[i], dtype=tf.float32)
                print('outputs shape:', outputs[0].get_shape())
                print('states shape:', states[0].get_shape()) # (batch_size, n_hidden)
                final_states.append(states)

                # Linear activation, using rnn inner loop
                final_output = [tf.sigmoid(tf.matmul(output, self.weights_out[i]) + self.biases_out[i]) for output in outputs]
                print('each final_outputs shape: ', final_output[0].get_shape())
                print('length of final_outputs: ', len(final_output))
                x = final_output

        final_output = tf.pack(final_output)
        print(final_output.get_shape())

        return [final_output, final_states]

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
