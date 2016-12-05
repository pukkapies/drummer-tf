import numpy as np
import soundfile
import json
from models.utilFunctions import window_dictionary
from models.stft import stftAnal ,stftSynth
from models.dftModel import dftAnal, dftSynth
from plotting import spectogram_plot
import matplotlib.pyplot as plt
from utils.generation_utils import unnormalise_range


wav_file = './preprocess/data/dataset_d_4_p_60.wav'
network_settings_file = './training/saved_models/stft_blackman_harris_800/network_settings.json'
# Load the network output (normalised)
mX_model_norm = np.load('./xtmag.npy')
pX_model_norm = np.load('./xtphase.npy')

print(np.max(mX_model_norm), np.min(mX_model_norm))

# Read the parameters from the settings json
with open(network_settings_file) as json_file:
    json_dict = json.load(json_file)
analysis_type = json_dict['analysis_type']
analysis_settings = json_dict[analysis_type + '_settings']

N = analysis_settings['N']
M = analysis_settings['M']
H = analysis_settings['H']
window = analysis_settings['w']
w = window_dictionary[window](M)
mag_range = analysis_settings['mag_range']
phase_range = analysis_settings['phase_range']
mag_normalised_range = analysis_settings['mag_normalised_range']
phase_normalised_range = analysis_settings['phase_normalised_range']

# mX_model = unnormalise_range(mX_model_norm, mag_normalised_range, mag_range, constrain_range=True)
# pX_model = unnormalise_range(pX_model_norm, phase_normalised_range, phase_range, constrain_range=True)

mX_model = unnormalise_range(mX_model_norm, [np.min(mX_model_norm), np.max(mX_model_norm)], mag_range, constrain_range=True)
pX_model = unnormalise_range(pX_model_norm, phase_normalised_range, phase_range, constrain_range=True)


# Load the wav
wav, sr = soundfile.read(wav_file)
mX, pX = stftAnal(wav, w, N, H)

print('wav file mX min/max: ', [np.min(mX), np.max(mX)])
print('model mX min/max: ', [np.min(mX_model), np.max(mX_model)])

print(mX_model[0, :])


print(mX.shape, mX_model.shape)
asdfasdf
plt.figure()
plt.subplot(2,1,1)
plt.plot(mX[10, :])
plt.subplot(2,1,2)
plt.plot(mX_model[10, :])
plt.show()