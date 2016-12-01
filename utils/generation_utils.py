import models.stft as STFT
import plotting
import os
from utils.sampling import binary_sample


class NetworkOutputProcessing(object):
    def __init__(self, network_output, network_settings):
        self.result = network_output
        self.settings = network_settings
        self.output_shape = network_output.shape
        self.check_network_output()
        self.convert_network_output_to_analysis_model_input()

    def convert_network_output_to_analysis_model_input(self):
        raise NotImplementedError

    def check_network_output(self):
        raise NotImplementedError

    def make_plots(self, waveform, w, M, N, H, sr, filepath=None):
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
        self.mX, self.pX = STFT.stftAnal(waveform, w, N, H)
        plotting.spectogram_plot(self.mX, self.pX, M, N, H, sr, show=False, filepath=filepath + 'model_spectogram')


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

        # Unnormalise
        self.xtfreq = freq_range[0] + (self.xtfreq * (freq_range[1] - freq_range[0]))
        self.xtphase = phase_range[0] + (self.xtphase * (phase_range[1] - phase_range[0]))
        self.xtmag = mag_range[0] + (self.xtmag * (mag_range[1] - mag_range[0]))

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
        assert self.result.shape[1] == 2 * self.n_freqs

    def convert_network_output_to_analysis_model_input(self):
        self.xtmag = self.result[:, :self.n_freqs]
        self.xtphase = self.result[:, self.n_freqs : 2*self.n_freqs]

        assert self.xtmag.shape == self.xtphase.shape

        phase_range = self.settings['stft_settings']['phase_range']
        mag_range = self.settings['stft_settings']['mag_range']

        # Unnormalise
        self.xtphase = phase_range[0] + (self.xtphase * (phase_range[1] - phase_range[0]))
        self.xtmag = mag_range[0] + (self.xtmag * (mag_range[1] - mag_range[0]))

        return self.xtmag, self.xtphase
