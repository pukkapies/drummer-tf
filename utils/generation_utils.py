from __future__ import print_function, absolute_import
import models.stft as STFT
import plotting
import numpy as np
import matplotlib.pyplot as plt

import os
from utils.sampling import binary_sample


def unnormalise_range(array, normalised_range, unnormalised_range, constrain_range=True):
    """
    Takes an array which is supposed to have values that lie within normalised_range, and scales
    the elements so that they lie within unnormalised_range. If constrain_range is True, then any
    values that lie outside the range are snapped to the boundary of the range.
    :param array: np.array, usually network output
    :param normalised_range: List [float, float]. The supposed range of values in array
    :param unnormalised_range: List [float, float]. The range of values we want array to have after rescaling
    :param constrain_range: Bool. Whether or not to fix outlying values to the boundary of the range
    :return: rescaled array
    """
    print('array min/max: ', [np.min(array), np.max(array)])
    rescaled_array = (array - normalised_range[0]) / (normalised_range[1] - normalised_range[0])  # between 0 and 1
    print('rescaled_array min/max: ', [np.min(rescaled_array), np.max(rescaled_array)])
    if constrain_range:
        rescaled_array[rescaled_array < 0.0] = 0.0
        rescaled_array[rescaled_array > 1.0] = 1.0
    rescaled_array = unnormalised_range[0] + (rescaled_array * (unnormalised_range[1] - unnormalised_range[0]))
    return rescaled_array


class NetworkOutputProcessing(object):
    def __init__(self, network_output, network_settings):
        self.result = network_output
        self.settings = network_settings
        self.output_shape = network_output.shape
        self.check_network_output()

    def convert_network_output_to_analysis_model_input(self):
        raise NotImplementedError

    def check_network_output(self):
        raise NotImplementedError

    def make_plots(self, waveform, w, M, N, H, sr, filepath=None):
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
        self.mX, self.pX = STFT.stftAnal(waveform, w, N, H)
        plotting.spectogram_plot(self.mX, self.pX, M, N, H, sr, show=False, filepath=filepath + 'model_generation_spectogram')




class SineModelOutputProcessing(NetworkOutputProcessing):
    def check_network_output(self):
        self.max_sines = self.settings['sine_model_settings']['max_sines']
        assert self.result.shape[1] == 3 * self.max_sines

    def convert_network_output_to_analysis_model_input(self):
        """
        Takes the output from the network and undoes the transformation applied to normalise
        :param xtfreq: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
        :param xtmag: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
        :param xtphase: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
        :param sinemodel_settings: dict that is saved in the json file with info on the transformation applied
        :return: The untransformed arrays xtfreq, xtmag, xtphase
        """
        self.xtfreq = self.result[:, :self.max_sines]
        self.xtmag = self.result[:, self.max_sines : 2*self.max_sines]
        self.xtphase = self.result[:, 2*self.max_sines : 3*self.max_sines]

        assert self.xtfreq.shape == self.xtmag.shape == self.xtphase.shape

        phase_range = self.settings['sine_model_settings']['phase_range']
        freq_range = self.settings['sine_model_settings']['freq_range']
        mag_range = self.settings['sine_model_settings']['mag_range']

        mag_normalised_range = self.settings['sine_model_settings']['mag_normalised_range']
        phase_normalised_range = self.settings['sine_model_settings']['phase_normalised_range']
        freq_normalised_range = self.settings['sine_model_settings']['freq_normalised_range']

        # Unnormalise
        self.xtfreq = unnormalise_range(self.xtfreq, freq_normalised_range, freq_range)
        self.xtphase = unnormalise_range(self.xtphase, phase_normalised_range, phase_range)
        self.xtmag = unnormalise_range(self.xtmag, mag_normalised_range, mag_range)

        return self.xtfreq, self.xtmag, self.xtphase

    def make_plots(self, waveform, w, M, N, H, sr, filepath=None):
        super(SineModelOutputProcessing, self).make_plots(waveform, w, M, N, H, sr, filepath)
        plotting.plot_sineTracks(self.mX, self.pX, M, N, H, sr, self.xtfreq, show=False,
                                 filepath=filepath + 'model_sinetracks')


class SineModelOutputProcessingWithActiveTracking(SineModelOutputProcessing):
    def check_network_output(self):
        self.max_sines = self.settings['sine_model_settings']['max_sines']
        assert self.result.shape[1] == 4 * self.max_sines

    def convert_network_output_to_analysis_model_input(self):
        self.active_tracks = self.result[:, 3 * self.max_sines: 4 * self.max_sines]

        super(SineModelOutputProcessingWithActiveTracking, self).convert_network_output_to_analysis_model_input()
        assert self.xtfreq.shape == self.xtmag.shape == self.xtphase.shape == self.active_tracks.shape
        sampled_active_tracks = binary_sample(self.active_tracks)
        self.xtfreq *= sampled_active_tracks

        return self.xtfreq, self.xtmag, self.xtphase


class STFTModelOutputProcessing(NetworkOutputProcessing):
    def check_network_output(self):
        self.n_freqs = (self.settings['stft_settings']['N'] // 2) + 1
        assert self.result.shape[1] == 2 * self.n_freqs, "array passed in has shape[1] = {}, but 2*n_freqs = {}".format(
            self.result.shape[1], 2*self.n_freqs
        )

    def convert_network_output_to_analysis_model_input(self):
        print('<<<< CONVERTING NETWORK OUTPUT >>>>')
        self.xtmag = self.result[:, :self.n_freqs]
        self.xtphase = self.result[:, self.n_freqs : 2*self.n_freqs]

        # plt.figure()
        # plt.plot(self.xtmag[1, :])
        # plt.show()

        assert self.xtmag.shape == self.xtphase.shape
        assert self.xtmag.shape[1] + self.xtphase.shape[1] == self.result.shape[1]

        phase_range = self.settings['stft_settings']['phase_range']
        mag_range = self.settings['stft_settings']['mag_range']

        mag_normalised_range = self.settings['stft_settings']['mag_normalised_range']
        phase_normalised_range = self.settings['stft_settings']['phase_normalised_range']

        print('mag norm range: {}'.format(mag_normalised_range))
        # Unnormalise
        print('mag max: {}'.format(np.max(self.xtmag)))
        print('mag min: {}'.format(np.min(self.xtmag)))

        print('phase max: {}'.format(np.max(self.xtphase)))
        print('phase min: {}'.format(np.min(self.xtphase)))

        self.xtmag = unnormalise_range(self.xtmag, mag_normalised_range, mag_range, constrain_range=False)
        self.xtphase = unnormalise_range(self.xtphase, phase_normalised_range, phase_range, constrain_range=False)

        print("After un-normalising: ")
        print('mag max: {}'.format(np.max(self.xtmag)))
        print('mag min: {}'.format(np.min(self.xtmag)))

        print('phase max: {}'.format(np.max(self.xtphase)))
        print('phase min: {}'.format(np.min(self.xtphase)))

        return self.xtmag, self.xtphase


