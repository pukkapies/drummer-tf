import numpy as np
import argparse
from datetime import datetime
from generation.generate_from_saved_model import main

SAMPLES = 16000
MODEL_FOLDER = './training/saved_models/'
SAVE_EVERY = None
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
WAV_OUT_PATH = './generation-' + STARTED_DATESTRING


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError('Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='LSTM model generation script')
    parser.add_argument(
        '--model_folder', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=WAV_OUT_PATH,
        help='Path to output wav file')
    parser.add_argument('--vector_folder', type=str, default=None,
                        help='The directory containing the vectorised data.')
    # parser.add_argument(
    #     '--wav_seed',
    #     type=str,
    #     default=None,
    #     help='The wav file to start generation from')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)


