import numpy as np
import soundfile
import json
from models.utilFunctions import window_dictionary
from models.stft import stftAnal ,stftSynth
from models.dftModel import dftAnal, dftSynth
from plotting import spectogram_plot
import matplotlib.pyplot as plt
import matplotlib
from utils.generation_utils import unnormalise_range
import os

# Test to get plots to dynamically update - the following works
#
# plt.ion()
#
# def get_new_data():
#     x = np.arange(10)
#     y = x**2 + np.random.rand(10)
#     print(x.shape, y.shape)
#     return x, y
#
# fig, ax = plt.subplots()
# ln, = ax.plot([], [], 'go-')
# while True:
#     x, y = get_new_data()
#     X, Y = ln.get_xdata(), ln.get_ydata()
#     ln.set_data(np.r_[X, x], np.r_[Y, y])
#     plt.draw()
#     plt.pause(0.001)


MODEL_NAME = 'stft_1500_M512_SHORT_TEST20'
vector_folder = './preprocess/stft/vectors/dataset_d_4_p_60_SHORT_TEST20/0/'

mX = np.load(vector_folder + 'mag.npy')
pX = np.load(vector_folder + 'phase.npy')

print('Data min/max: ', np.min(mX), np.max(mX))

network_output_folder = './generation/network_output/{}/'.format(MODEL_NAME)
if not os.path.exists(network_output_folder):
    os.makedirs(network_output_folder)

# Load the network output (normalised)
mX_model = np.load(network_output_folder + 'xtmag_model.npy')
pX_model = np.load(network_output_folder + 'xtphase_model.npy')

print(mX_model.shape)
print(pX_model.shape)


# wav_file = './preprocess/data/dataset_d_4_p_60.wav'
network_settings_file = './training/saved_models/{}/network_settings.json'.format(MODEL_NAME)

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


# Unnormalise data
mX = unnormalise_range(mX, mag_normalised_range, mag_range)
pX = unnormalise_range(pX, phase_normalised_range, phase_range)

# mX_model = unnormalise_range(mX_model_norm, mag_normalised_range, mag_range, constrain_range=True)
# pX_model = unnormalise_range(pX_model_norm, phase_normalised_range, phase_range, constrain_range=True)

# mX_model = unnormalise_range(mX_model_norm, [np.min(mX_model_norm), np.max(mX_model_norm)], mag_range, constrain_range=True)
# pX_model = unnormalise_range(pX_model_norm, phase_normalised_range, phase_range, constrain_range=True)


# Load the wav
# wav, sr = soundfile.read(wav_file)
# mX, pX = stftAnal(wav, w, N, H)

print('data mX min/max: ', [np.min(mX), np.max(mX)])
print('model mX min/max: ', [np.min(mX_model), np.max(mX_model)])

for i in range(10):

    # if 100 % (i+1) != 0: continue

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title('Frame {}'.format(i))
    plt.plot(mX[i, :])
    plt.subplot(4, 1, 2)
    plt.plot(mX_model[i, :])

    # print('N: {}, M: {}'.format(N, M))
    # print(mX_model[i, :].size)

    frame_i = dftSynth(mX[i, :], pX[i, :], M)
    frame_i_model = dftSynth(mX_model[i, :], pX[i, :], M)


    plt.subplot(4,1,3)
    plt.plot(frame_i)
    plt.subplot(4,1,4)
    plt.plot(frame_i_model)

    plt.show()

    # print(np.sum((mX[0, :] - mX_model[0, :])**2)/mX[1, :].size)

