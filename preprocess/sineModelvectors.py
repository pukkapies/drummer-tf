import models.sineModel as sineModel
import models.utilFunctions as UF
from models.utilFunctions import nextbiggestpower2, window_dictionary
import soundfile
import os
from scipy.signal import blackmanharris
import numpy as np
import json
import plotting
import models.stft as STFT
import warnings


FOLDER_LIST = ['./data']
OUTPUT_FOLDER = './vectors'
PLOT_FOLDER = './plots'
SAMPLE_RATE = 44100 # Assume all files to be loaded have the same sample rate, or raise an error

# Sinusoidal model parameters
MAX_N_SINES = 100
THRESHOLD = -90 # threshold for sinusoidal amplitudes

# STFT parameters
N = 4096
M = 2047
H = 512
window = 'blackmanharris'

w = window_dictionary.get(window, None)(M)

class InvalidPathError(Exception): pass



def load_npy(filepath):
    if not os.path.exists(filepath):
        raise InvalidPathError("{} does not exist!".format(filepath))
    xtfreq = np.load(filepath + '/freq.npy')
    xtmag = np.load(filepath + '/mag.npy')
    xtphase = np.load(filepath + '/phase.npy')
    active_tracks = np.load(filepath + '/active_tracks.npy')
    return xtfreq, xtmag, xtphase, active_tracks

def load_from_dir_root(rootdir):
    if not os.path.exists(rootdir):
        raise InvalidPathError("{} does not exist!".format(rootdir))
    if not os.path.exists(rootdir + '/SineModel_settings.json'):
        raise Exception("Sinemodel_settings.json file not found. Maybe the data hasn't been vectorised yet.")
    with open(rootdir + '/SineModel_settings.json') as json_file:
        json_vector_settings_dict = json.load(json_file)
    loaded_data = []
    for root, dir, filenames in os.walk(rootdir):
        if all(x in filenames for x in ['freq.npy', 'mag.npy', 'phase.npy', 'active_tracks.npy']):
            print('Loading files from {}...'.format(root), end='')
            xtfreq, xtmag, xtphase, active_tracks = load_npy(root)
            print('done')
            loaded_data.append([xtfreq, xtmag, xtphase, active_tracks])
    return loaded_data, json_vector_settings_dict

def create_json(settings_file, json_dict):
    with open(settings_file, 'w') as json_file:
        json.dump(json_dict, json_file)

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    file_count = 0
    datalist = []

    json_settings_file = OUTPUT_FOLDER + '/SineModel_settings.json'
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

            # All outputs of sineModelAnal are of the shape (numFrames, maxSines)
            xtfreq, xtmag, xtphase = sineModel.sineModelAnal(file, sr, w, N, H, THRESHOLD, maxnSines=MAX_N_SINES)

            active_tracks = (xtfreq!=0.0).astype(int)

            # For plotting the spectrogram of the signal
            mX, pX = STFT.stftAnal(file, w, N, H)
            plotting.plot_sineTracks(mX, pX, M, N, H, sr, xtfreq, show=False,
                                                     filepath=PLOT_FOLDER + '/{}_sinetracks'.format(audio_file[:-4]))
            plotting.spectogram_plot(mX, pX, M, N, H, sr, show=False, filepath=PLOT_FOLDER + '/{}'.format(audio_file[:-4]))

            # Process the frequencies, magnitudes and phases to be normalised in the range [0,1]
            xtfreq = xtfreq / freq_range[1]

            #TODO: we might want to calculate this across all of the training data instead of file by file
            #TODO: This will need modifying in the json file as well
            min_xtmag = np.min(xtmag)
            max_xtmag = np.max(xtmag[xtmag != 0]) # Recall tracks are separated by zeros - we want to ignore them
            xtmag[xtmag == 0] = min_xtmag # Could change this later to have the dB floor lower for the zeros
            xtmag = (xtmag - min_xtmag) / (max_xtmag - min_xtmag)

            xtphase = np.mod(xtphase, 2 * np.pi) / (2 * np.pi)  # Between 0 and 1

            json_dict['mag_range'] = [min_xtmag, max_xtmag]

            assert (xtfreq <=1).all() and (xtfreq >= 0).all()
            assert (xtmag <= 1).all() and (xtmag >= 0).all()
            assert (xtphase <= 1).all() and (xtphase >= 0).all()

            output_path = OUTPUT_FOLDER + '/{}/'.format(file_count)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            ## Save the numpy arrays separately - couldn't work out how to save and load multiple arrays
            np.save(output_path + 'freq', xtfreq)
            np.save(output_path + 'mag', xtmag)
            np.save(output_path + 'phase', xtphase)
            np.save(output_path + 'active_tracks', active_tracks)

            datalist.append([xtfreq, xtmag, xtphase, active_tracks])
            print('Saved as {}'.format(output_path))
            file_count += 1

    create_json(json_settings_file, json_dict)

if __name__ == '__main__':
    main()