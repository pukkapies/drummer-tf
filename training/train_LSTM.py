import preprocess.sineModelvectors as vecs
import tensorflow as tf
import json
import time
import os
from nn_models.rnn_models import setup_non_self_updating_rnn, SimpleLSTM
# from nn_models.rnn_models import setup_self_updating_rnn
from .setup.setup_data import setup_training_data
from .seq2seq_rnn_models import RNN_Model
from .training_utils import Patience


# Network Parameters
n_steps = 164 # timesteps
n_hidden = [1000] # hidden layer num of features in LSTM
# n_hidden2 = 1000 # hidden layer of features in second LSTM
n_outputs = 100 * 3 # output prediction
n_input = 100 * 3   # 100 frequencies, amplitudes, phases

json_settings = {'n_steps': n_steps,
                 'hidden_units1': n_hidden,
                 'n_outputs': n_outputs,
                 'n_inputs': n_input}

def load_saved_model_to_resume_training(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir), end='')

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def main(args):
    # loaded is a list of lists. Each sublist is length 3, with np.array entries of xtfreq, xtmag, xtphase
    # Each shape is (164, 100) = (numFrames, maxSines)
    # json_vector_settings is a dict with settings used for the SineModel
    loaded, json_vector_settings = vecs.load_from_dir_root(args.vector_folder)
    json_settings['SineModel_settings'] = json_vector_settings

    NUM_TRAINING_FILES = len(loaded)

    placeholders, data_dict = setup_training_data(loaded, args.batch_size)

    y = placeholders['output_data']
    x = placeholders['input_data']
    feed_dict = {y: data_dict['output_data'], x: data_dict['input_data']}

    lstm = SimpleLSTM(x, None, n_hidden, n_outputs)
    pred = lstm.prediction


    # pred = setup_non_self_updating_rnn(x, n_hidden[0], n_outputs)
    print('pred shape: ', pred.get_shape())
    print('y shape:', y.get_shape())

    print('y - pred shape:', (y-pred).get_shape())

    cost = tf.nn.l2_loss(pred - y) / tf.cast(tf.size(pred - y), tf.float32)
    tf.scalar_summary('cost', cost)


    patience = Patience(args)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      args.grad_clip)
    optimizer = tf.train.AdamOptimizer(args.learning_rates[patience.learning_rates_index]).apply_gradients(zip(grads, tvars))

    # Set up logging for TensorBoard.
    writer = tf.train.SummaryWriter(args.model_folder)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.merge_all_summaries()

    # Initializing the variables

    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:

        sess.run(init)
        print([var.name for var in tf.all_variables()])

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        step = load_saved_model_to_resume_training(saver, sess, args.model_folder)
        if step == None: # Couldn't find a checkpoint to restore from
            step = 0

        if args.model_folder[-1] != '/':
            args.model_folder = args.model_folder + '/' # Make sure logging info is saved correctly

        with open(args.model_folder + 'network_settings.json', mode='w') as settings_file:
            json.dump(json_settings, settings_file)

        while step < args.num_training_steps:

            start_time = time.time()

            # print(sess.run(grads, feed_dict=feed_dict_for_training_vars))

            summary, _ = sess.run([summaries, optimizer], feed_dict=feed_dict)
            writer.add_summary(summary, step)

            # Calculate batch loss
            loss = sess.run(cost, feed_dict=feed_dict)

            # Check patience
            patience_reached, new_best_cost = patience.update(loss)

            if new_best_cost: # and (patience.learning_rates_index + 1 == len(args.learning_rates)):
                # Save the model for a new best cost and delete the old saved model
                print('New best cost achieved, saving the model... ', end='')
                for file in os.listdir(args.model_folder):
                    if 'best_cost_model' in file:
                        os.remove(file)
                saver.save(sess, args.model_folder + '/best_cost_model', global_step=step)
                print('done.')
            if patience_reached:
                if patience.learning_rates_index + 1 > len(args.learning_rates): #Ran out of learning rates, so terminate
                    print('Max patience reached...')
                    print("Iter " + str(step * args.batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss))
                    print('Best model cost = {}'.format(patience.best_cost))
                    break
                else:
                    print("Patience reached, new learning rate = {}".format(args.learning_rates[patience.learning_rates_index]))

            if step % args.display_step == 0:
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}, patience = {}, time for minibatch: {}".format(loss, patience.iterations,
                                                                             time.time() - start_time))
                saver.save(sess, args.model_folder, global_step=step)
            if step % args.save_every == 0:
                print('Saving the model... ', end='')
                saver.save(sess, args.model_folder, global_step=step)
                print('done.')
            if step % 1000 == 0:
                print('evaluating model... predicted followed by ground truth:')
                print(sess.run([pred, y], feed_dict=feed_dict))
            step += 1

        print("Maximum iterations reached... optimization Finished!")
        print(sess.run([pred, y], feed_dict=feed_dict))
