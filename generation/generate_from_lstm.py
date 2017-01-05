from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
import os
from models.utilFunctions import nextbiggestpower2, window_dictionary
from models.sineModel import sineModelSynth
from models.stft import stftSynth, stftAnal
import soundfile
from nn_models.rnn_models import SimpleLSTM
from datetime import datetime
from utils.utils import load_saved_model_to_resume_training
from utils.generation_utils import SineModelOutputProcessing, SineModelOutputProcessingWithActiveTracking
from utils.generation_utils import STFTModelOutputProcessing
from utils.vectorisation_utils import load_from_dir_root
from training.setup.setup_data import setup_training_data
import matplotlib.pyplot as plt
from plotting import spectogram_plot
from nn_models.layers import Dense
from nn_models.initialisers import wbVars_Xavier

STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())


def main(args):
    # Verify model folder
    if not os.path.exists(args.model_folder):
        raise Exception("Model folder does not exist!")

    # Get network settings
    settings_json_string = '/network_settings.json'
    if args.model_folder[-1] == '/':
        args.model_folder = args.model_folder[:-1]

    if os.path.isfile(args.model_folder):
        model_folder = os.path.join(*args.model_folder.split(sep='/')[:-1])
    else:
        model_folder = args.model_folder

    model_name = model_folder.split(sep='/')[-1]

    with open(model_folder + settings_json_string, 'r') as f:
        network_settings = json.load(f)

    analysis_type = network_settings['analysis_type']
    analysis_settings = network_settings[analysis_type + '_settings']
    M = analysis_settings['M']
    H = analysis_settings['H']
    w = window_dictionary.get(analysis_settings['w'])(M)
    N = analysis_settings['N']
    sr = analysis_settings['sample_rate']

    input_data_shape = (1, 1, network_settings['n_inputs'])  # (num_steps, batch_size, n_inputs)


    with tf.Session() as sess:

        n_hidden = network_settings['n_hidden']  # List of hidden unit sizes

        input_placeholder = tf.placeholder(tf.float32, shape=input_data_shape)
        input = np.zeros(input_data_shape)
        feed_dict = {input_placeholder: input}

        state_placeholders = []
        states = []
        for lstm_layer in range(len(n_hidden)):
            state_placeholders.append((tf.placeholder(tf.float32, shape=(1, n_hidden[lstm_layer]), name='cell'),  # batch_size = 1
                                       tf.placeholder(tf.float32, shape=(1, n_hidden[lstm_layer]), name='hidden')))  # batch_size = 1
            states.append([np.zeros((1, n_hidden[lstm_layer])), np.zeros((1, n_hidden[lstm_layer]))])
            feed_dict[state_placeholders[lstm_layer]] = states[lstm_layer]

        print('n_steps:', network_settings['n_steps'])

        n_outputs = network_settings['n_outputs']
        if len(n_hidden) == 1:
            n_outputs = [n_outputs]
        else:
            n_outputs = n_hidden[1:] + [n_outputs]

        x = input_placeholder
        lstm_states = []  # Keep track of LSTM states
        for i in range(len(n_hidden)):
            lstm = SimpleLSTM(n_hidden[i], scope='LSTM_model/layer_{}'.format(i + 1))
            lstm_output, state = lstm(x, state_placeholders[i])  # lstm_outputs is Tensor of shape (n_steps, batch_size, n_hidden[i])
            lstm_states.append(state)
            lstm_output = tf.unpack(lstm_output)  # Make it into list length n_steps, each entry (batch_size, n_hidden[i])
            dense = Dense(scope="LSTM_model/layer_{}".format(i + 1), size=n_outputs[i],
                                   nonlinearity=tf.sigmoid, initialiser=wbVars_Xavier)
            final_output = dense(lstm_output[0])

        saver = tf.train.Saver()
        # saver.restore(sess, './training/saved_models/2016-11-25T08-54-35/model-20')
        load_saved_model_to_resume_training(saver, sess, model_folder)

        outputs_list = []
        for step in range(network_settings['n_steps']):
            output, *states = sess.run([final_output] + [state for state in lstm_states],
                                      feed_dict=feed_dict)  # states is list of LSTMStateTuple (length num_layers)
            # output is shape (batch_size, n_outputs), but it needs to be (n_steps=1, batch_size, n_outputs)
            output = np.expand_dims(output, axis=0)
            outputs_list.append(output)
            lstm_layer_states= states
            input = output
            # Update feed_dict by giving the new input and states for all layers
            feed_dict[input_placeholder] = input
            for lstm_layer in range(len(n_hidden)):
                feed_dict[state_placeholders[lstm_layer]] = lstm_layer_states[lstm_layer]

        final_outputs = [tf.squeeze(output, [0]) for output in outputs_list]
        final_outputs = tf.pack(final_outputs)  # Make one tensor of rank 2
        print(final_outputs.get_shape())
        final_outputs = tf.transpose(final_outputs,
                                     [1, 0, 2])  # final_outputs has shape (batch_size, n_frames, n_outputs)
        print('after packing and transposing, final_outputs shape: ', final_outputs.get_shape())

        ####################
        # Compare to ground truth (debugging)
        if args.vector_folder is None:
            print("No vector folder was provided, cannot calculate cost for debugging.")
        else:
            loaded, json_vector_settings, analysis_type_check = load_from_dir_root(args.vector_folder)
            assert analysis_type==analysis_type_check

            _, data_dict = setup_training_data(loaded, 1)  # batch_size = 1
            ground_truth = data_dict['output_data']  # (n_frames, batch_size, n_outputs)
            ground_truth = np.transpose(ground_truth, [1, 0, 2])
            print('Network generation: {}'.format(final_outputs.eval()))
            network_mag = final_outputs.eval()[0, :, :257]
            # plt.figure()
            # plt.subplot(3,1,1)
            # plt.plot(network_mag[0, :])
            # plt.subplot(3,1,2)
            # plt.plot(network_mag[1, :])
            # plt.subplot(3,1,3)
            # plt.plot(network_mag[3, :])
            # plt.show()

            print(network_mag[1, 7], network_mag[1, 17], network_mag[1, 32])


            ground_truth_mag = ground_truth[0, :, :257]
            # plt.figure()
            # plt.subplot(3, 1, 1)
            # plt.plot(ground_truth_mag[0, :])
            # plt.subplot(3, 1, 2)
            # plt.plot(ground_truth_mag[1, :])
            # plt.subplot(3, 1, 3)
            # plt.plot(ground_truth_mag[2, :])
            # plt.show()


            print('Data: {}'.format(ground_truth))
            print('Network output min/max: {}, {}'.format(np.min(final_outputs.eval()), np.max(final_outputs.eval())))
            print('Data min/max: {}, {}'.format(np.min(ground_truth), np.max(ground_truth)))
            print(ground_truth.shape)
            print(final_outputs.eval().shape)
            assert ground_truth.shape == final_outputs.eval().shape
            print('Squared error achieved by network: {}'.format(np.sum((ground_truth - final_outputs.eval())**2)
                                                                 / ground_truth.size))

            # mX = final_outputs.eval()[0,:,:257]
            # pX = final_outputs.eval()[0,:,257:]
            #
            # np.save('./mX_model', mX)
            # np.save('./pX_model', pX)
            #
            # asdfasdf
            # phase_range = network_settings['stft_settings']['phase_range']
            # mag_range = network_settings['stft_settings']['mag_range']
            #
            # # Unnormalise
            # mX = mag_range[0] + (mX * (mag_range[1] - mag_range[0]))
            # mX *= sr / 2  # Undo weird rescaling
            # pX = phase_range[0] + (pX * (phase_range[1] - phase_range[0]))
            #
            # print(mX.shape)  # (num_frames, num_freq_bins)
            # # mX = np.transpose(mX)
            # # pX = np.transpose(pX)
            # spectogram_plot(mX, pX, M, N, H, sr, fig_number=None, filepath=None, show=True)
            #
            # reconst = stftSynth(mX, pX, M, H)
            # print(np.max(reconst), np.min(reconst))
            #
            #
            # # soundfile.write('./test_reconst.wav', reconst, sr, format='wav')
            # mX2, pX2 = stftAnal(reconst, w, N, H)
            # print(mX2.shape, pX2.shape)
            # spectogram_plot(mX2, pX2, M, N, H, sr, fig_number=None, filepath=None, show=True)
            # asdfasdfasd


        ####################
        # For testing - just batch size = 1
        result = tf.squeeze(final_outputs, [0]).eval()

    network_output_folder = './generation/network_output/{}/'.format(model_name)
    if not os.path.exists(network_output_folder):
        os.makedirs(network_output_folder)

    if analysis_type == 'sine_model':
        process_output = SineModelOutputProcessingWithActiveTracking(result, network_settings)
        xtfreq, xtmag, xtphase = process_output.convert_network_output_to_analysis_model_input()
        reconstruction = sineModelSynth(xtfreq, xtmag, xtphase, nextbiggestpower2(analysis_settings['M']), H, sr)
    elif analysis_type == 'sine_model_without_active_tracking':  # deprecated
        process_output = SineModelOutputProcessing(result, network_settings)
        xtfreq, xtmag, xtphase = process_output.convert_network_output_to_analysis_model_input()
        reconstruction = sineModelSynth(xtfreq, xtmag, xtphase, nextbiggestpower2(analysis_settings['M']), H, sr)
    elif analysis_type == 'stft':
        process_output = STFTModelOutputProcessing(result, network_settings)
        xtmag, xtphase = process_output.convert_network_output_to_analysis_model_input()
        np.save(network_output_folder + 'xtmag_model', xtmag)
        np.save(network_output_folder + 'xtphase_model', xtphase)
        plot_filepath = './generation/plots/{}-(generated_{})'.format(model_name, STARTED_DATESTRING)
        if not os.path.exists(plot_filepath): os.makedirs(plot_filepath)
        spectogram_plot(xtmag, xtphase, M, N, H, sr, show=False, filepath=plot_filepath + '/network_output_spectogram')
        reconstruction = stftSynth(xtmag, xtphase, M, H)
    else:
        raise Exception('analysis_type not recognised!')

    print('model_name:', model_name)

    #TODO: extract more of these arguments in the class methods
    process_output.make_plots(reconstruction, w, M, N, H, sr,
               filepath='./generation/plots/{}-(generated_{})/'.format(model_name, STARTED_DATESTRING))

    soundfile.write('./generation/wav_output/{}-(generated_{}).wav'.format(model_name, STARTED_DATESTRING),
                    reconstruction, sr, format='wav')


