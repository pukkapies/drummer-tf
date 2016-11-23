import tensorflow as tf
from training.rnn_models import colah_lstm
import numpy as np

colah_lstm_n_hidden = 40

training_sequence = np.array([[[-0.2, 0.111],
                              [0.54, 0.111],
                              [-0.03, -0.6],
                               [0.222, 0.53],
                               [0.8, -0.423]]]) # (batch_size, n_steps, n_outputs)

training_seq_shape = training_sequence.shape
num_steps = training_seq_shape[1]
n_inputs = training_seq_shape[0]

y = tf.placeholder("float", shape=training_seq_shape)

colah_lstm_cell = colah_lstm(n_inputs, colah_lstm_n_hidden, 1, 'colah_lstm') # n_inputs, n_hidden, batch_size
hidden_weights = tf.Variable(tf.random_normal([colah_lstm_n_hidden, 2]))
hidden_biases = tf.Variable(tf.zeros(2))

final_outputs = []

output = tf.zeros((1,2))
for step in range(num_steps):
    output = colah_lstm_cell.update(output)
    output = tf.tanh(tf.matmul(output, hidden_weights) + hidden_biases)
    final_outputs.append(output)

print('outputs length: ', len(final_outputs))
print('outputs shape:', final_outputs[0].get_shape()) # (batch_size, n_hidden)

final_outputs = tf.pack(final_outputs)  # Make one tensor of rank 2
final_outputs = tf.transpose(final_outputs, [1, 0, 2])  # final_outputs has shape (batch_size, n_frames, n_hidden)
print('after packing and transposing, final_outputs shape: ', final_outputs.get_shape())

cost = tf.nn.l2_loss(final_outputs - y) # / tf.cast(tf.size(final_outputs - y), tf.float32)
tf.scalar_summary('cost', cost)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

writer = tf.train.SummaryWriter('./testfolder/')
writer.add_graph(tf.get_default_graph())
summaries = tf.merge_all_summaries()

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)
    print([var.name for var in tf.all_variables()])

    print('initial outputs: ', sess.run(final_outputs))
    print('initial cost: ', sess.run(cost, feed_dict={y : training_sequence}))

    # test the model
    test_final_outputs = []
    test_output = tf.zeros((1, 2))
    for _ in range(3):
        test_output = colah_lstm_cell.update(test_output)
        test_output = tf.matmul(test_output, hidden_weights) + hidden_biases
        test_final_outputs.append(test_output)
    print("Testing the model....")
    print(final_outputs.eval())

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    for step in range(1000):
        summary, _ = sess.run([summaries, optimizer], feed_dict={y : training_sequence})

        loss = sess.run(cost, feed_dict={y : training_sequence})
        saver.save(sess, './testfolder/', global_step=step)
        if step % 10 == 0:
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
        if step % 100 == 0:
            print("Testing the model....")
            print(final_outputs.eval())

    # test the model
    test_final_outputs = []
    test_output = tf.zeros((1, 2))
    for step in range(3):
        test_output = colah_lstm_cell.update(test_output)
        test_output = tf.matmul(test_output, hidden_weights) + hidden_biases
        test_final_outputs.append(test_output)
    print("Testing the model....")
    print(final_outputs.eval())





