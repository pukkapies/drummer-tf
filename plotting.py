import matplotlib.pyplot as plt
import numpy as np
import math
from models.sineModel import extracttracks


def spectogram_plot(mX, pX, M, N, H, sr, fig_number=None, filepath=None, show=True):
    """
    Plots the spectogram for STFT and saves the file under filepath.
    :param mX: Magnitude spectrum. Numpy matrix of size (num_of_frames, num_of_frequency_bins)
    :param pX: Phase spectrum. Numpy matrix of size (num_of_frames, num_of_frequency_bins)
    :param M: Window size (int)
    :param N: FFT size (int)
    :param H: Hop size (int)
    :param sr: Sample rate
    :param fig_number: Figure number
    :param filepath: filepath string for saving the file. If None, the file isn't saved.
    :param show: Boolean, whether to show the plot.
    :return: None
    """
    assert (N/2 + 1) == mX.shape[1] == pX.shape[1]

    if fig_number:
        fig = plt.figure(fig_number, figsize=(9.5, 6))
    else:
        fig = plt.figure(figsize=(9.5, 6))

    if filepath:
        filename = filepath.split(sep='/')[-1]
    else:
        filename = ''

    ax1 = fig.add_subplot(2, 1, 1)
    numFrames = int(mX[:,0].size)
    frmTime = H * np.arange(numFrames) / float(sr)  # Times in seconds corresponding to the centre of each window
    binFreq = np.arange(N/2+1) * float(sr) / N  # Frequencies of each FFT bin
    # plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
    plt.pcolormesh(np.transpose(mX))
    plt.title('mX ({}), M={}, N={}, H={}'.format(filename, M, N, H))
    plt.autoscale(tight=True)

    ax2 = fig.add_subplot(2, 1, 2)
    numFrames = int(pX[:,0].size)
    frmTime = H * np.arange(numFrames) / float(sr)
    binFreq = np.arange(N/2+1) * float(sr) / N
    # plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(pX), axis=0))
    plt.pcolormesh(np.transpose(pX))
    plt.title('pX difference ({}), M={}, N={}, H={}'.format(filename, M, N, H))
    plt.autoscale(tight=True)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath + '.png')
    if show:
        plt.show()
    return fig, ax1, ax2

def dft_plot(x, w, mX, pX, sr, fig_number=None, filepath='dft.png', show=True):
    """
    Plots the signal x together with its DFT under the window w. Saves under the filepath.
    :param x: Original signal, restricted to the support of the window. Numpy array of shape (window_length, )
    :param w: Window. Numpy array of shape (window_length, )
    :param mX: Magnitude spectrum. Numpy array of shape ((FFT_size / 2) + 1, )
    :param pX: Phase spectrum. Numpy array of shape ((FFT_size / 2) + 1, )
    :param sr: Sampling rate (int)
    :param fig_number: Figure number
    :param filepath: filepath string for saving the file. If None, the file isn't saved.
    :param show: Boolean, whether to show the plot.
    :return:
    """

    assert x.shape==w.shape

    filename = filepath.split(sep='/')[-1]

    N = 2 * (mX.size - 1) # FFT size

    hM1 = int(math.floor(
        (w.size + 1) / 2))  # Num samples in first half of window (includes centre sample for odd size window)
    hM2 = int(math.floor(w.size / 2))  # Num samples in second half of window

    if fig_number:
        fig = plt.figure(fig_number, figsize=(9.5, 7))
    else:
        fig = plt.figure(figsize=(9.5, 7))

    ax1 = fig.add_subplot(3, 1, 1)
    plt.plot(np.arange(-hM1, hM2), x, lw=1.5)
    plt.axis([-hM1, hM2, min(x), max(x)])
    plt.ylabel('amplitude')
    plt.title('x ({})'.format(filename))

    ax2 = fig.add_subplot(3, 1, 2)
    plt.plot(sr * np.arange(mX.size) / N, mX, 'r', lw=1.5)
    plt.axis([0, sr * mX.size / N, min(mX), max(mX)])

    plt.title('magnitude spectrum: mX = 20*log10(abs(X))')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude (dB)')

    ax3 = fig.add_subplot(3, 1, 3)
    plt.plot(sr * np.arange(mX.size) / N, pX, 'c', lw=1.5)
    plt.axis([0, sr * mX.size / N, min(pX), max(pX)])
    plt.title('phase spectrum: pX=unwrap(angle(X))')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('phase (radians)')

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath + '.png')
    if show:
        plt.show()
    return fig, ax1, ax2, ax3

def plot_peaks(peak_locs, sr, mX, pX, fig_number, filepath='dft_with_peaks.png', show=True):
    """
    Adds peak markers to an existing DFT plot in Figure fig_number (plot should not yet be shown).
    Saves under filepath.
    :param peak_locs: 1D numpy array of peak locations in Hz
    :param sr: Sample rate (int)
    :param mX: Magnitude spectrum. Numpy array of shape ((FFT_size / 2) + 1, )
    :param pX: Phase spectrum. Numpy array of shape ((FFT_size / 2) + 1, )
    :param fig_number: Figure number of a DFT plot to add peak markers to
    :param filepath: filepath string for saving the file. If None, the file isn't saved.
    :param show: Boolean, whether to show the plot.
    :return:
    """

    N = 2 * (mX.size - 1)  # FFT size
    pmag = mX[peak_locs]
    plt.figure(fig_number)
    plt.subplot(312)
    plt.plot(sr * peak_locs / float(N), pmag, marker='x', linestyle='')

    plt.subplot(313)
    plt.plot(sr * peak_locs / float(N), pX[peak_locs], marker='x', linestyle='')

    if filepath:
        plt.savefig(filepath + '.png')
    if show:
        plt.show()


def plot_sineTracks(mX, pX, M, N, H, sr, xtfreq, fig_number=None, filepath=None, show=True):
    """
    Adds sinusoidal tracks to an existing spectogram plot in Figure fig_number (plot should not yet be shown).
    Saves under filepath.
    :param xtfreq: numpy array shape (num_Frames, max_Sines) with sine frequencies as entries, tracks separated by zero
    :param sr: Sample rate, int
    :param H: Hop size, int
    :param fig_number: Figure number of a DFT plot to add peak markers to
    :param filepath: filepath string for saving the file. If None, the file isn't saved.
    :param show: Boolean, whether to show the plot.
    :return:
    """
    print('inside plot_sineTracks, mX.shape = ', mX.shape)
    if fig_number:
        fig = plt.figure(fig_number, figsize=(9.5, 6))
    else:
        fig = plt.figure(figsize=(9.5, 6))

    if filepath:
        filename = filepath.split(sep='/')[-1]

    ax1 = fig.add_subplot(2, 1, 1)
    numFrames = int(mX[:, 0].size)
    print('numFrames', numFrames)
    frmTime = H * np.arange(numFrames) / float(sr)  # Times in seconds corresponding to the centre of each window
    binFreq = np.arange(N / 2 + 1) * float(sr) / N  # Frequencies of each FFT bin
    print(xtfreq.shape)

    for track_seq in range(xtfreq.shape[1]):
        trackFreqs = xtfreq[:, track_seq]  # frequencies of one track
        trackBegs, trackEnds = extracttracks(trackFreqs)
        for i, j in zip(trackBegs, trackEnds):
            plt.plot(frmTime[i: j], trackFreqs[i: j])

    plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
    plt.title('mX ({}), M={}, N={}, H={}'.format(filename, M, N, H))
    plt.autoscale(tight=True)

    ax2 = fig.add_subplot(2, 1, 2)
    numFrames = int(pX[:, 0].size)
    frmTime = H * np.arange(numFrames) / float(sr)
    binFreq = np.arange(N / 2 + 1) * float(sr) / N
    plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(pX), axis=0))
    plt.title('pX difference ({}), M={}, N={}, H={}'.format(filename, M, N, H))
    plt.autoscale(tight=True)

    plt.tight_layout()

    if filepath:
        plt.savefig(filepath + '.png')
    if show:
        plt.show()

    return fig, ax1, ax2