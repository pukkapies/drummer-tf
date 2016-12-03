from utils.vectorisation_utils import load_from_dir_root
import tensorflow as tf
import json
import time
import os
from nn_models.rnn_models import SimpleLSTM
from .setup.setup_data import setup_training_data
from utils.training_utils import Patience
from utils.utils import load_saved_model_to_resume_training


def main(args):
    # loaded is a list of lists. Each sublist is length 3, with np.array entries of xtfreq, xtmag, xtphase
    # Each shape is (164, 100) = (numFrames, maxSines)
    # json_vector_settings is a dict with settings used for the SineModel
    loaded, json_vector_settings, analysis_type = load_from_dir_root(args.vector_folder)

    if not os.path.exists(args.model_folder):
        print("Model folder does not exist, training new model.")
        os.makedirs(args.model_folder)

    assert isinstance(loaded, list)
    n_hidden = args.lstm_hidden_units

    num_layers = len(n_hidden)
    initial_states = [None] * num_layers

    n_steps = loaded[0][0].shape[0]
    n_input = sum([loaded[0][i].shape[1] for i in range(len(loaded[0]))])

    n_outputs = n_input

    json_settings = {'n_hidden': n_hidden,
                     'n_outputs': n_outputs,
                     'n_inputs': n_input,
                     'n_steps': n_steps,
                     analysis_type + '_settings': json_vector_settings,
                     'analysis_type': analysis_type}

    placeholders, data_dict = setup_training_data(loaded, args.batch_size)

    y = placeholders['output_data']
    x = placeholders['input_data']
    feed_dict = {y: data_dict['output_data'], x: data_dict['input_data']}
    lr_placeholder = tf.placeholder(tf.float32) # Learning rate

    lstm = SimpleLSTM(x, initial_states, n_hidden, n_outputs, activation_fn=tf.sigmoid)
    pred = lstm.prediction

    # pred = setup_non_self_updating_rnn(x, n_hidden[0], n_outputs)
    print('pred shape: ', pred.get_shape()) # (batch_size, n_steps, 4*max_sines)
    print('y shape:', y.get_shape())

    print('y - pred shape:', (y-pred).get_shape())

    cost = tf.nn.l2_loss(pred - y) / tf.cast(tf.size(pred - y), tf.float32)
    tf.scalar_summary('cost', cost)

    patience = Patience(args)

    feed_dict[lr_placeholder] = args.learning_rates[patience.learning_rates_index]

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      args.grad_clip)
    optimizer = tf.train.AdamOptimizer(lr_placeholder)
    optimize = optimizer.apply_gradients(zip(grads, tvars))

    print([var.name for var in tf.all_variables()])

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

        step, model_folder = load_saved_model_to_resume_training(saver, sess, args.model_folder)
        if step == None:  # Couldn't find a checkpoint to restore from
            step = 0

        # if model_folder.split(sep='/')[-1] == 'best_model':
        #     # best_model_folder = model_folder
        #     model_folder = os.path.join(*(model_folder.split(sep='/')[:-1]))
        #     print("Using best model folder. best_model_folder = {}, model_folder = {}".format(best_model_folder, model_folder))
        # else:
        #     best_model_folder = model_folder + 'best_model/'
        #
        # if not os.path.exists(best_model_folder):
        #     os.makedirs(best_model_folder)

        with open(model_folder + 'network_settings.json', mode='w') as settings_file:
            json.dump(json_settings, settings_file)
        # with open(best_model_folder + 'network_settings.json', mode='w') as settings_file:
        #     json.dump(json_settings, settings_file)

        while step < args.num_training_steps:
            start_time = time.time()

            last_saved_at_step = 0

            summary, _ = sess.run([summaries, optimize], feed_dict=feed_dict)
            writer.add_summary(summary, step)

            # Calculate batch loss
            loss = sess.run(cost, feed_dict=feed_dict)

            # Check patience
            patience_reached, new_best_cost, plateau_reached = patience.update(loss)
            print('Iteration {}, best_cost = {}'.format(patience.iterations, new_best_cost))

            if patience_reached:
                print('patience reached')
                if patience.learning_rates_index + 1 > len(args.learning_rates): #Ran out of learning rates, so terminate
                    print('Max patience reached...')
                    print("Iter " + str(step * args.batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss))
                    print('Best model cost = {}'.format(patience.best_cost))
                    break
                else:
                    print("Patience reached, new learning rate = {}".format(args.learning_rates[patience.learning_rates_index]))
                    feed_dict[lr_placeholder] = args.learning_rates[patience.learning_rates_index]

            if new_best_cost and step - last_saved_at_step >= args.save_every:
                # Save the model for a new best cost and delete the old saved model
                # print('New best cost achieved, saving the model... ', end='')
                # for file in os.listdir(best_model_folder):
                #     if 'best_cost_model' in file:
                #         os.remove(best_model_folder + file)
                # saver.save(sess, best_model_folder + 'best_cost_model', global_step=step)
                saver.save(sess, model_folder + 'model', global_step=step)
                last_saved_at_step = step
                # print('done.')

            if step % args.display_step == 0:
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}, patience = {}, time for minibatch: {}".format(loss, patience.iterations,
                                                                             time.time() - start_time))
            # if step % args.save_every == 0:
            #     print('Saving the model... ', end='')
            #     saver.save(sess, model_folder + 'model', global_step=step)
            #     print('done.')
            if step % 1000 == 0:
                print('evaluating model... predicted followed by ground truth:')
                print(sess.run([pred, y], feed_dict=feed_dict))
            step += 1

        print("Maximum iterations reached... optimization Finished!")
        print(sess.run([pred, y], feed_dict=feed_dict))
        saver.save(sess, model_folder + 'model', global_step=step)

        # git_label = subprocess.check_output(["git", "describe"])
        json_settings['best_cost'] = float(patience.best_cost)
        # json_settings['git_commit'] = git_label
        with open(model_folder + 'network_settings.json', mode='w') as settings_file:
            json.dump(json_settings, settings_file)
        # with open(best_model_folder + 'network_settings.json', mode='w') as settings_file:
        #     json.dump(json_settings, settings_file)

