# from __future__ import print_function, absolute_import
import sys

sys.path.insert(0,'.')

import argparse
from utils.vectorisation_utils import load_from_dir_root
from utils.dataset import DatasetFeed
import os
from nn_models.ae_encoders import AE_LSTMEncoder
from nn_models.ae_decoders import AE_LSTMDecoder
from nn_models.autoencoder import Autoencoder
import tensorflow as tf
from utils.vectorisation_utils import create_json
from datetime import datetime
import json


VECTOR_FOLDER = None

# Training parameters
LEARNING_RATE_LIST = [0.001, 0.0001, 0.00001]
NUM_TRAINING_STEPS = 10000
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
DISPLAY_STEP = 10
SAVE_EVERY = 500
CHECKPOINT_EVERY = 500
MAX_PATIENCE = 500
PLATEAU_TOL = (5000, 0.0001)  # Parameters for detecting plateau. (n_iterations, minimum_decrease_in_cost)

N_HIDDEN = [1000] # hidden layer num of features in LSTM

# If it's not specified in the arguments then a datestamped subfolder will be created for the model
MODEL_FOLDER = './training/saved_models'


def get_arguments():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='LSTM audio synth model')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='How many wav files to process at once.')
    parser.add_argument('--vector_folder', type=str, default=VECTOR_FOLDER,
                        help='The directory containing the vectorised data.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard.')
    parser.add_argument('--model_folder', type=str, default=None,
                        help='Directory in which to store/restore the model')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_training_steps', type=int, default=NUM_TRAINING_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--num_batches_per_grad_update', type=int, default=1,
                        help='Number of minibatches to accumulate gradients before applying them')
    parser.add_argument('--learning_rates', default=LEARNING_RATE_LIST, type=float, nargs='+',
                        help='Learning rate list for training.')
    parser.add_argument('--lstm_encoder_hidden_units', default=N_HIDDEN, type=int, nargs='+',
                        help='Number of hidden units in each LSTM encoder layer')
    parser.add_argument('--lstm_decoder_hidden_units', default=N_HIDDEN, type=int, nargs='+',
                        help='Numbers of hidden units in each LSTM decoder layer')
    parser.add_argument('--display_step', type=int, default=DISPLAY_STEP,
                        help='How often to display training progress and save model.')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='Gradient clipping')
    parser.add_argument('--save_every', type=int, default=SAVE_EVERY,
                        help='How often to save the model.')
    parser.add_argument('--max_patience', type=int, default=MAX_PATIENCE,
                        help='Maximum number of iterations for patience.')
    parser.add_argument('--plateau_tol', type=float, nargs='+',
                        help='Plateau tolerance. Number of iterations and minimum cost decrease.')
    return parser.parse_args()


def main(args):
    loaded, json_vector_settings, analysis_type = load_from_dir_root(args.vector_folder)

    model_folder = args.model_folder

    batch_size = args.batch_size
    dataset = DatasetFeed(loaded, batch_size)

    dataset.set_all_data_blocks_to_max_shape()

    data_shape = dataset.max_data_shape  # Same for normal and reversed

    n_steps = data_shape[0]
    print('n_steps:', n_steps)
    n_outputs = data_shape[1]

    if os.path.exists(model_folder):  # Resume a previously trained model
        if not os.path.isfile(model_folder + ".meta"):
            raise ValueError("Model_folder exists, but supplied path is not a meta file (supply path without '.meta')")
        print("Model folder exists. Resuming training from {}".format(model_folder))
        meta_graph = model_folder
        model_folder = '/'.join(meta_graph.split('/')[:-1]) + '/'
        with open(model_folder + '/network_settings.json') as json_file:
            json_settings = json.load(json_file)
        n_input = json_settings['n_inputs']
        assert n_outputs == json_settings['n_outputs'], "Vector outputs size does not match that stored" \
                                                                     "in the network_settings.json file"
        log_dir = model_folder + 'log/'  # For tensorboard
        analysis_dir = model_folder + 'analysis/'  # For learning curves etc.
        for dir in [log_dir, analysis_dir]:
            if not os.path.exists(dir): os.makedirs(dir)
        autoencoder = Autoencoder(model_to_restore=meta_graph, d_hyperparams={},
                  log_dir=log_dir, analysis_dir=analysis_dir, json_dict=json_settings)
        autoencoder.dataset = dataset

        cost = autoencoder.train(max_iter=args.num_training_steps)
        json_settings['epochs_completed'] = autoencoder.dataset.epochs_completed
        json_settings['cost'] = float(cost)

        create_json(model_folder + 'network_settings.json', json_settings)
    else:  # Create a new model
        if model_folder[-1] != '/':
            model_folder += '/'
        print("Model folder does not exist, training new model.")
        os.makedirs(model_folder)

        n_hidden_encoder = args.lstm_encoder_hidden_units[0]  # Just one hidden layer for now
        n_hidden_decoder = args.lstm_decoder_hidden_units[0]

        encoder = AE_LSTMEncoder(n_hidden_encoder)
        decoder = AE_LSTMDecoder(n_hidden_decoder, n_outputs, n_steps=n_steps, output_activation=tf.sigmoid)

        n_input = n_outputs

        model_datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        json_settings = {'model_datetime': model_datetime,
                         'n_hidden_encoder': n_hidden_encoder,
                         'n_hidden_decoder': n_hidden_decoder,
                         'n_outputs': n_outputs,
                         'n_inputs': n_input,
                         analysis_type + '_settings': json_vector_settings,
                         'analysis_type': analysis_type}

        input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, None, n_input], name="input")
        # The following placeholder is for feeding the ground truth to the LSTM decoder - the first input should be zeros
        shifted_input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, None, n_input], name="shifted_input")
        print('input_placeholder shape: ', input_placeholder.get_shape())

        log_dir = model_folder + 'log/'  # For tensorboard
        analysis_dir = model_folder + 'analysis/'  # For learning curves etc.
        for dir in [log_dir, analysis_dir]:
            if not os.path.exists(dir): os.makedirs(dir)

        build_dict = {'encoder': encoder,
                      'decoder': decoder,
                      'n_input': n_input,
                      'input_placeholder': input_placeholder,
                      'shifted_input_placeholder': shifted_input_placeholder,
                      'dataset': dataset,
                      'model_folder': model_folder}

        autoencoder = Autoencoder(build_dict=build_dict, d_hyperparams={},
                  log_dir=log_dir, analysis_dir=analysis_dir, json_dict=json_settings)

        create_json(model_folder + 'network_settings.json', json_settings)

        cost = autoencoder.train(max_iter=args.num_training_steps)
        json_settings['epochs_completed'] = autoencoder.dataset.epochs_completed
        json_settings['cost'] = float(cost)
        json_settings['global_step'] = int(autoencoder.step)

        create_json(model_folder + 'network_settings.json', json_settings)


if __name__ == '__main__':

    args = get_arguments()

    if args.model_folder is None:
        args.model_folder = MODEL_FOLDER + '/' + STARTED_DATESTRING

    if args.vector_folder is None:
        raise Exception('No vectors folder specified. (Use --vector_folder argument)')
    else:
        if not os.path.exists(args.vector_folder):
            raise Exception('{} not found'.format(args.vector_folder))

    if args.batch_size is None:
        raise Exception('Need to specify the number of audio samples to process at each step (--batch_size)')

    if args.vector_folder[-1] == '/':
        args.vector_folder = args.vector_folder[:-1]

    main(args)
