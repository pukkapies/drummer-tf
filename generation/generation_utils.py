import models.stft as STFT
import plotting
import os


def convert_network_output_to_sinemodel_input(xtfreq, xtmag, xtphase, sinemodel_settings):
    """
    Takes the output from the network and undoes the transformation applied to normalise
    :param xtfreq: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
    :param xtmag: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
    :param xtphase: numpy array shape (n_frames, n_sines) normalised to be between 0 and 1
    :param sinemodel_settings: dict that is saved in the json file with info on the transformation applied
    :return: The untransformed arrays xtfreq, xtmag, xtphase
    """
    phase_range = sinemodel_settings['phase_range']
    freq_range = sinemodel_settings['freq_range']
    mag_range = sinemodel_settings['mag_range']

    xtfreq_untransformed = freq_range[0] + (xtfreq * (freq_range[1] - freq_range[0]))
    xtphase_untransformed = phase_range[0] + (xtphase * (phase_range[1] - phase_range[0]))
    xtmag_untransformed = mag_range[0] + (xtmag * (mag_range[1] - mag_range[0]))

    return xtfreq_untransformed, xtmag_untransformed, xtphase_untransformed


def make_plots(waveform, w, M, N, H, sr, xtfreq, filepath=None):
    if filepath:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    mX, pX = STFT.stftAnal(waveform, w, N, H)
    plotting.plot_sineTracks(mX, pX, M, N, H, sr, xtfreq, show=False,
                             filepath=filepath + 'model_sinetracks')
    plotting.spectogram_plot(mX, pX, M, N, H, sr, show=False, filepath=filepath + 'model_spectogram')