from __future__ import print_function, absolute_import

import models.utilFunctions as UF
from models.utilFunctions import nextbiggestpower2, window_dictionary
import soundfile
import os
import numpy as np
import json
import plotting
import models.stft as STFT
import warnings
from utils.vectorisation_utils import create_json, InvalidPathError

FOLDER_LIST = ['./data']
OUTPUT_FOLDER = './stft/vectors/TTMI02X01'
PLOT_FOLDER = './stft/plots'
SAMPLE_RATE = 44100 # Assume all files to be loaded have the same sample rate, or raise an error

# STFT parameters
N = 512
M = 511
H = 128
window = 'blackmanharris'

mX_norm_range = [0.1, 0.9]
pX_norm_range = [0.1, 0.9]

w = window_dictionary.get(window, None)(M)


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)

    file_count = 0
    datalist = []

    json_settings_file = OUTPUT_FOLDER + '/stft_settings.json'
    json_dict = {'N': N, 'M': M, 'H': H, 'w': window, 'phase_range': [0, 2 * np.pi]}

    print("Loading files in {}".format(FOLDER_LIST))
    for folder in FOLDER_LIST:
        if not os.path.exists(folder):
            raise InvalidPathError("{} not found".format(folder))
        file_list = os.listdir(folder)
        for audio_file in file_list:
            if audio_file[-3:] not in ['wav', 'aif']:
                print('Skipping {}'.format(audio_file))
                continue
            print("Processing {}...".format(audio_file), end='')
            file, sr = soundfile.read(folder + '/' + audio_file)
            freq_range = [0, sr / 2]
            if sr!=SAMPLE_RATE:
                warnings.warn('Sample rate is not 44100Hz')

            json_dict['sample_rate'] = sr
            json_dict['freq_range'] = freq_range

            mX, pX = STFT.stftAnal(file, w, N, H)

            # # For SHORT_TEST:
            # mX = mX[:20, :]
            # pX = pX[:20, :]

            # For plotting the spectrogram of the signal
            plotting.spectogram_plot(mX, pX, M, N, H, sr, show=False, filepath=PLOT_FOLDER + '/{}'.format(audio_file[:-4]))

            # Process the frequencies, magnitudes and phases to be normalised in the range [0,1]
            #TODO: we might want to calculate this across all of the training data instead of file by file
            #TODO: This will need modifying in the json file as well
            min_mX = np.min(mX)
            max_mX = np.max(mX)

            json_dict['mag_range'] = [min_mX, max_mX]

            json_dict['mag_normalised_range'] = mX_norm_range
            json_dict['phase_normalised_range'] = pX_norm_range

            mX = (mX - min_mX) / (max_mX - min_mX) # Between 0 and 1
            mX = (mX * (mX_norm_range[1] - mX_norm_range[0])) + mX_norm_range[0]
            pX = np.mod(pX, 2 * np.pi) / (2 * np.pi)  # Between 0 and 1
            pX = (pX * (pX_norm_range[1] - pX_norm_range[0])) + pX_norm_range[0]

            # Check the data has been normalised correctly
            assert (mX <= mX_norm_range[1]).all() and (mX >= mX_norm_range[0]).all()
            assert (pX <= pX_norm_range[1]).all() and (pX >= pX_norm_range[0]).all()

            output_path = OUTPUT_FOLDER + '/{}/'.format(file_count)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            ## Save the numpy arrays separately - couldn't work out how to save and load multiple arrays
            np.save(output_path + 'mag', mX)
            np.save(output_path + 'phase', pX)

            datalist.append([mX, pX])
            print('Saved as {}'.format(output_path))
            file_count += 1

    create_json(json_settings_file, json_dict)

if __name__ == '__main__':
    main()