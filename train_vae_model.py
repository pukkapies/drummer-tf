# from __future__ import print_function, absolute_import
import sys

sys.path.insert(0,'.')

import argparse
from datetime import datetime
import os
from training.train_vae import main


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
    parser.add_argument('--samples_per_batch', type=int, default=1,
                        help='Number of Gaussian samples for each element in the batch.')
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
    parser.add_argument('--deterministic_warm_up', type=int, default=0,
                        help='Number of training steps of "deterministic warm up", where KL cost is removed')
    parser.add_argument('--learning_rates', default=LEARNING_RATE_LIST, type=float, nargs='+',
                        help='Learning rate list for training.')
    parser.add_argument('--lstm_encoder_hidden_units', default=N_HIDDEN, type=int, nargs='+',
                        help='Number of hidden units in each LSTM encoder layer')
    parser.add_argument('--prelatent_dense_layers', default=[], type=int, nargs='+',
                        help='List of hidden layer sizes to feed into latent mean/stdev')
    parser.add_argument('--lstm_decoder_hidden_units', default=N_HIDDEN, type=int, nargs='+',
                        help='Numbers of hidden units in each LSTM decoder layer')
    parser.add_argument('--postlatent_dense_layers', default=[], type=int, nargs='+',
                        help='Numbers of hidden units to feed into cell and hidden states of decoder')
    parser.add_argument('--latent_space_dimension', default=2, type=int,
                         help='Dimension of the latent (z) space')
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
