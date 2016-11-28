import argparse
from datetime import datetime
import os
from training.train_LSTM import main


VECTOR_FOLDER = "./preprocess/sine_model/vectors/dataset_d_4_p_60"

# Training parameters
LEARNING_RATE_LIST = [0.001, 0.0001, 0.00001]
NUM_TRAINING_STEPS = 10000
BATCH_SIZE = 3
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
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once.')
    parser.add_argument('--vector_folder', type=str, default=VECTOR_FOLDER,
                        help='The directory containing the vectorised data.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard.')
    parser.add_argument('--model_folder', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_training_steps', type=int, default=NUM_TRAINING_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rates', default=LEARNING_RATE_LIST, type=float, nargs='+',
                        help='Learning rate list for training.')
    parser.add_argument('--lstm_hidden_units', default=N_HIDDEN, type=int, nargs='+',
                        help='Number of hidden units in each LSTM layer')
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
    # parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
    #                     help='JSON file with the network parameters.')
    # parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
    #                     help='Concatenate and cut audio samples to this many '
    #                     'samples.')
    # parser.add_argument('--l2_regularization_strength', type=float,
    #                     default=L2_REGULARIZATION_STRENGTH,
    #                     help='Coefficient in the L2 regularization. '
    #                     'Disabled by default')
    # parser.add_argument('--silence_threshold', type=float,
    #                     default=SILENCE_THRESHOLD,
    #                     help='Volume threshold below which to trim the start '
    #                     'and the end from the training set samples.')
    # parser.add_argument('--optimizer', type=str, default='adam',
    #                     choices=optimizer_factory.keys(),
    #                     help='Select the optimizer specified by this option.')
    # parser.add_argument('--momentum', type=float,
    #                     default=MOMENTUM, help='Specify the momentum to be '
    #                     'used by sgd or rmsprop optimizer. Ignored by the '
    #                     'adam optimizer.')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_arguments()

    if args.model_folder is None:
        args.model_folder = MODEL_FOLDER + '/' + STARTED_DATESTRING

    args.plateau_tol[0] = int(args.plateau_tol[0])
    args.plateau_tol = tuple(args.plateau_tol)

    if not os.path.exists(VECTOR_FOLDER):
        raise Exception('{} not found'.format(VECTOR_FOLDER))

    main(args)
