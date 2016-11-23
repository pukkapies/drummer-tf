import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import resample, blackmanharris

import models.dftModel as DFT
import models.stft as STFT
import models.sineModel as sineModel
import plotting
import models.utilFunctions as UF
from models.utilFunctions import nextbiggestpower2


# filepath = "./rock_kit/TTMI02X07.aif"
filepath = "./rock_kit/TTMI02X01.aif"
filename = filepath.split(sep='/')[-1]

save_path = "./example_data/"

x, sr = soundfile.read(filepath)

print('file lenght = ', x.size)

#########################################################
#                                                       #
#                   DFT Settings                        #
#                                                       #
#########################################################


M = 2047  # Window size
w = blackmanharris(M)    # USE 75% OVERLAP FOR BLACKMAN-HARRIS

N = 4096 # FFT size
H = 512 # Hop size

hM1 = int(math.floor((w.size+1)/2))  # Num samples in first half of window (includes centre sample for odd size window)
hM2 = int(math.floor(w.size/2))  # Num samples in second half of window

# # print('Computing DFT...')
# pin = 5000
# x1 = x[pin-hM1:pin+hM2]
# mX, pX = DFT.dftAnal(x1, w, N)
# print('DFT done.')

# plotting.dft_plot(x1, w, mX, pX, sr, fig_number=1, filename=filename, show=True)

# peak_locs = UF.peakDetection(mX, -70)
# plotting.plot_peaks(peak_locs, sr, mX, pX, fig_number=1, filename=filename + '_peaks')

#############################


# print('Computing spectrogram...')
# mX, pX = STFT.stftAnal(x, w, N, H)
# print('Spectogram done.')
#
# # Compute reconstruction
# y = STFT.stftSynth(mX, pX, M, H)
# soundfile.write(save_path + '{}_reconstruction(N={}_M={}_H={}).wav'.format(filename[:-4], N, M, H), y, sr, format='wav')

# Compare plots of origin and reconstruction
# plt.subplot(2,1,1)
# plt.plot(x[:40000])
# plt.subplot(2,1,2)
# plt.plot(y[:40000])
# plt.show()

# spec_fig, spec_ax1, spec_ax2 = plotting.spectogram_plot(mX, pX, M, N, H, sr, fig_number=1, show=False)

############################


# Sinusoidal model

maxnSines = 100
t = -90 # threshold for sinusoidal amplitudes
print('Computing sinusoidal tracks...')
xtfreq, xtmag, xtphase = sineModel.sineModelAnal(x, sr, w, N, H, t, maxnSines=maxnSines)
print("sine model shapes: ", xtfreq.shape, xtmag.shape, xtphase.shape)
print('Sinusoidal tracks done.')

numFrames = int(xtfreq[:,0].size)
frmTime = H * np.arange(numFrames) / float(sr)
# xtfreq[xtfreq<=0] = np.nan
sineModel_reconst = sineModel.sineModelSynth(xtfreq, xtmag, xtphase, nextbiggestpower2(M), H, sr)

print('sinemodel reconstruction size:', sineModel_reconst.size)

soundfile.write(save_path + '{}_sineModel_reconstruction_thresh={}_{}sines_N={}_M={}_H={}.wav'.format
                (filename[:-4], np.abs(t), maxnSines, N, M, H),
                sineModel_reconst, sr, format='wav')
# plotting.plot_sineTracks(xtfreq, sr, H, 1, show=False)
# plt.show()

asdfasdfa

############################

plt.figure()
plt.subplot(2,1,1)
plt.plot(x)
plt.subplot(2,1,2)
plt.plot(sineModel_reconst)
# plt.show()

############################

# testarr = np.arange(51) - 25
# lobe = UF.genBhLobe(testarr)
# plt.plot(lobe)
# plt.show()

# # NB 21963.8671875 = (44100 / 2) - (44100/ / N) : It is the frequency of the second-to-last frequency bin
# x = UF.genSpecSines_p(np.array([21963.8671875]), np.array([-0.6]), np.array([1.1]), 512, 44100)
# plt.plot(np.abs(x))
# plt.show()