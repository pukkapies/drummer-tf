import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.python.ops import rnn, rnn_cell
from models.utilFunctions import nextbiggestpower2, window_dictionary
from models.sineModel import sineModelSynth
import soundfile
import models.stft as STFT
import plotting
from nn_models.rnn_models import SimpleLSTM
from datetime import datetime
from utils.utils import load_saved_model_to_resume_training

STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())


def convert_network_output_to_sinemodel_input(xtfreq, xtmag, xtphase, sinemodel_settings):
    """
    Takes the output from the network and undoes the transformation applied to normalise
    :param xtfreq: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
    :param xtmag: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
    :param xtphase: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
    :param sinemodel_settings: dict that is saved in the json file with info on the transformation applied
    :return: The untransformed arrays xtfreq, xtmag, xtphase
    """
    phase_range = sinemodel_settings['phase_range']
    freq_range = sinemodel_settings['freq_range']
    mag_range = sinemodel_settings['mag_range']

    xtfreq_untransformed = freq_range[0] + (xtfreq * (freq_range[1] - freq_range[0]))
    xtphase_untransformed = phase_range[0] + (xtphase * (phase_range[1] - phase_range[0]))
    xtmag_untransformed = mag_range[0] + (xtmag * (mag_range[1] - mag_range[0]))

    return xtfreq_untransformed, xtmag_untransformed, xtphase_untransformed

def make_plots(waveform, w, M, N, H, sr, xtfreq, filepath=None):
    if filepath:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    mX, pX = STFT.stftAnal(waveform, w, N, H)
    plotting.plot_sineTracks(mX, pX, M, N, H, sr, xtfreq, show=False,
                             filepath=filepath + 'model_sinetracks')
    plotting.spectogram_plot(mX, pX, M, N, H, sr, show=False, filepath=filepath + 'model_spectogram')


def main(args):

    # Verify model folder
    if not os.path.exists(args.model_folder):
        raise Exception("Model folder does not exist!")

    # Get network settings
    settings_json_string = '/network_settings.json'
    if args.model_folder[-1] == '/':
        args.model_folder = args.model_folder[:-1]

    if os.path.isfile(args.model_folder):
        enclosing_model_folder = os.path.join(*args.model_folder.split(sep='/')[:-1])
    else:
        enclosing_model_folder = args.model_folder

    with open(enclosing_model_folder + settings_json_string, 'r') as f:
        network_settings = json.load(f)

    sinemodel_settings = network_settings['SineModel_settings']
    M = sinemodel_settings['M']
    H = sinemodel_settings['H']
    w = window_dictionary.get(sinemodel_settings['w'])(M)
    N = sinemodel_settings['N']
    sr = sinemodel_settings['sample_rate']

    input_data_shape = (1, 1, network_settings['n_inputs'])


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
            states.append((np.zeros((1, n_hidden[lstm_layer])), np.zeros((1, n_hidden[lstm_layer]))))
            feed_dict[state_placeholders[lstm_layer]] = states[lstm_layer]

        lstm = SimpleLSTM(input_placeholder, state_placeholders, n_hidden, network_settings['n_outputs'])

        print([var.name for var in tf.all_variables()])
        saver = tf.train.Saver()

        # saver.restore(sess, './training/saved_models/2016-11-25T08-54-35/model-20')
        load_saved_model_to_resume_training(saver, sess, enclosing_model_folder)

        outputs_list = []

        print('n_steps:', network_settings['n_steps'])

        for step in range(network_settings['n_steps']):
            # output, state = rnn.rnn(lstm.cell, input, initial_state=state, dtype=tf.float32)
            output, state = sess.run([lstm.prediction, lstm.state],
                                     feed_dict=feed_dict)
            outputs_list.append(output)
            input = output
            # Update feed_dict by giving the new input and states for all layers
            feed_dict[input_placeholder] = input
            for lstm_layer in range(len(n_hidden)):
                feed_dict[state_placeholders[lstm_layer]] = state[lstm_layer]

        final_outputs = [tf.squeeze(output, [0]) for output in outputs_list]
        final_outputs = tf.pack(final_outputs)  # Make one tensor of rank 2
        print(final_outputs.get_shape())
        final_outputs = tf.transpose(final_outputs,
                                     [1, 0, 2])  # final_outputs has shape (batch_size, n_frames, n_hidden)
        print('after packing and transposing, final_outputs shape: ', final_outputs.get_shape())

        ####################
        # For testing - just batch size = 1
        result = tf.squeeze(final_outputs, [0]).eval()

    xtfreq = result[:, :100]
    xtmag = result[:, 100:200]
    xtphase = result[:, 200:]

    print(xtfreq.shape, xtmag.shape, xtphase.shape)

    xtfreq, xtmag, xtphase = convert_network_output_to_sinemodel_input(xtfreq, xtmag, xtphase, sinemodel_settings)

    print(xtfreq.shape, xtmag.shape, xtphase.shape)

    # NB note that the reconstructed model is probably a bit shorter than the original, because of the hop size
    # not exactly dividing the signal length
    sineModel_reconst = sineModelSynth(xtfreq, xtmag, xtphase, nextbiggestpower2(sinemodel_settings['M']), H, sr)

    print(args.model_folder)

    print(args.model_folder.split(sep='/'))
    model_name = args.model_folder.split(sep='/')[-1]
    print('model_name:', model_name)

    make_plots(sineModel_reconst, w, M, N, H, sr, xtfreq,
               filepath='./generation/plots/{}-(generated_{})/'.format(model_name, STARTED_DATESTRING))

    soundfile.write('./generation/wav_output/{}-(generated_{}).wav'.format(model_name, STARTED_DATESTRING),
                    sineModel_reconst, sinemodel_settings['sample_rate'], format='wav')
