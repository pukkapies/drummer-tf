import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class RNN_Model():
    def __init__(self, n_input, model, rnn_size, rnn_num_layers, n_outputs, batch_size, input_seq_length,
                 grad_clip, infer=True):

        if infer:
            batch_size = 1
            input_seq_length = 1

        if model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(model))

        cell = cell_fn(rnn_size, state_is_tuple=True)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * rnn_num_layers, state_is_tuple=True)

        self.n_input = n_input
        self.input_data = tf.placeholder(tf.int32, [batch_size, input_seq_length, n_input])
        self.targets = tf.placeholder(tf.int32, [batch_size, input_seq_length, n_outputs])
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnn_model'):
            output_w = tf.get_variable("output_w", [rnn_size, n_outputs])
            output_b = tf.get_variable("output_b", [n_outputs])

            inputs = tf.split(1, input_seq_length, self.input_data)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, output_w) + output_b
            return prev

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                  loop_function=loop if infer else None, scope='rnn_model')

        #The following gives (batch_size * input_seq_length, rnn_size) shape tensor
        output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
        self.logits = tf.matmul(output, output_w) + output_b # (batch_size * input_seq_length, n_outputs)
        self.output = tf.sigmoid(self.logits)
        self.loss = tf.reduce_sum((tf.reshape(self.targets, [-1, n_outputs]) - self.output)**2) / \
               (batch_size * input_seq_length * n_outputs)

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, num, state=None):
        if state is None:
            state = sess.run(self.cell.zero_state(1, tf.float32))

        starting_input = np.zeros((1, self.n_input))

        outputs = []
        input = starting_input
        for n in range(num):
            feed = {self.input_data: input, self.initial_state: state}
            [output, state] = sess.run([self.output, self.final_state], feed)
            output.append(output)
            input = output
        return outputs, state