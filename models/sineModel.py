# functions that implement analysis and synthesis of sounds using the Sinusoidal Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft, fftshift
import math
import models.dftModel as DFT
import models.utilFunctions as UF
from models.utilFunctions import isPower2, nextbiggestpower2
import warnings


def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
    """
    Tracking sinusoids from one frame to the next
    pfreq, pmag, pphase: frequencies and magnitude of current frame
    tfreq: frequencies of incoming tracks from previous frame
    freqDevOffset: minimum frequency deviation at 0Hz
    freqDevSlope: slope increase of minimum frequency deviation
    returns tfreqn, tmagn, tphasen: frequency (Hz), magnitude and phase of tracks. Those tracks in tfreq that are
    continuing into the present frame appear in tfreqn, tmagn, tphasen in the same index position as they were
    in tfreq. New tracks are placed into tfreqn, tmagn, tphasen where tfreq had empty indices (so that it is easy
    to distinguish different tracks - there will also be at least one zero before/after the track).
    tfreqn, tmagn, tphasen are increased in size if necessary (with new tracks ordered by decreasing magnitude).
    """

    tfreqn = np.zeros(tfreq.size)  # initialize array for output frequencies
    tmagn = np.zeros(tfreq.size)  # initialize array for output magnitudes
    tphasen = np.zeros(tfreq.size)  # initialize array for output phases
    pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]  # indexes of current peaks
    incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0]  # indexes of incoming tracks
    newTracks = np.zeros(tfreq.size, dtype=np.int) - 1  # initialize to -1 new tracks
    magOrder = np.argsort(-pmag[pindexes])  # order current peaks by magnitude
    pfreqt = np.copy(pfreq)  # copy current peaks to temporary array
    pmagt = np.copy(pmag)  # copy current peaks to temporary array
    pphaset = np.copy(pphase)  # copy current peaks to temporary array

    # continue incoming tracks - for each peak (from highest to lowest), look for a frequency in tfreq (incoming freqs)
    # that is sufficiently close. Then newTracks stores the information for which tracks will continue. It is -1 for
    # all tracks that will not continue, otherwise it saves the index of the freq in pfreq that continues it.
    if incomingTracks.size > 0:  # if incoming tracks exist
        for i in magOrder:  # iterate over current peaks
            if incomingTracks.size == 0:  # break when no more incoming tracks
                break
            track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))  # closest incoming track to peak
            freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]])  # measure freq distance
            if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small
                newTracks[incomingTracks[track]] = i  # assign peak index to track index
                incomingTracks = np.delete(incomingTracks, track)  # delete index of track in incoming tracks
    indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]  # indexes of previous frame tracks that are continuing
    if indext.size > 0:
        indexp = newTracks[indext]  # indexes of continuing peaks in tfreq
        tfreqn[indext] = pfreqt[indexp]  # output freq tracks
        tmagn[indext] = pmagt[indexp]  # output mag tracks
        tphasen[indext] = pphaset[indexp]  # output phase tracks
        pfreqt = np.delete(pfreqt, indexp)  # delete used peaks
        pmagt = np.delete(pmagt, indexp)  # delete used peaks
        pphaset = np.delete(pphaset, indexp)  # delete used peaks

    # create new tracks from non used peaks
    emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]  # indexes of empty incoming tracks
    peaksleft = np.argsort(-pmagt)  # sort peaks that are left in current frame by magnitude
    # fill empty tracks of tfreqn (that were also empty for tfreq) with all remaining current peaks:
    if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):
        tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
        tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
        tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
    elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):  # add more tracks if necessary
        tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
        tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
        tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
        tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
        tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
        tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
    return tfreqn, tmagn, tphasen


def cleaningSineTracks(tfreq, minTrackLength=3):
    """
    Delete short fragments of a collection of sinusoidal tracks
    tfreq: frequency of tracks
    minTrackLength: minimum duration of tracks in number of frames
    returns tfreqn: output frequency of tracks
    """

    if tfreq.shape[1] == 0:  # if no tracks return input
        return tfreq
    nFrames = tfreq[:, 0].size  # number of frames
    nTracks = tfreq[0, :].size  # number of tracks in a frame
    for t in range(nTracks):  # iterate over all tracks
        trackFreqs = tfreq[:, t]  # frequencies of one track
        trackBegs, trackEnds = extracttracks(trackFreqs)

        trackLengths = trackEnds - trackBegs  # lengths of track contours
        for i, j in zip(trackBegs, trackLengths):  # delete short track contours
            if j <= minTrackLength:
                trackFreqs[i:i + j] = 0
    return tfreq

def extracttracks(trackFreqs):
    """
    Extracts the sinusoidal tracks from a sequence of frequencies, where the tracks are separated by zero
    :param trackFreqs: 1D numpy array
    :return: Two 1D arrays of indices corresponding to track beginnings and track endings
    NB: The track endings indices are the indices of the zero just after the track has finished, unless the track
    goes to the end, in which case it is the index of the last entry.
    """
    assert trackFreqs.ndim == 1

    nFrames = trackFreqs.size

    # Get the indices of the track beginnings
    trackBegs = np.nonzero((trackFreqs[:nFrames - 1] <= 0)
                           & (trackFreqs[1:] > 0))[0] + 1
    if trackFreqs[0] > 0:  # Check separately if there is a track starting at the beginning
        trackBegs = np.insert(trackBegs, 0, 0)

    # Get the indices of the track ends
    trackEnds = np.nonzero((trackFreqs[:nFrames - 1] > 0)  # end of track contours
                           & (trackFreqs[1:] <= 0))[0] + 1
    if trackFreqs[nFrames - 1] > 0:
        trackEnds = np.append(trackEnds, nFrames) # NB This index is 1 after the end of the array

    return trackBegs, trackEnds


def sineModel(x, fs, w, N, t):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB
    returns y: output array sound
    """

    hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    Ns = 512  # FFT size for synthesis (even)
    H = Ns / 4  # Hop size used for analysis and synthesis
    hNs = Ns / 2  # half of synthesis FFT size
    pin = max(hNs, hM1)  # init sound pointer in middle of anal window
    pend = x.size - max(hNs, hM1)  # last sample to start a frame
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    yw = np.zeros(Ns)  # initialize output sound frame
    y = np.zeros(x.size)  # initialize output array
    w = w / sum(w)  # normalize analysis window
    sw = np.zeros(Ns)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hNs - H:hNs + H] = ow  # add triangular window
    bh = blackmanharris(Ns)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window
    while pin < pend:  # while input sound pointer is within sound
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect locations of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        # -----synthesis-----
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)  # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))  # compute inverse FFT
        yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
        yw[hNs - 1:] = fftbuffer[:hNs + 1]
        y[pin - hNs:pin + hNs] += sw * yw  # overlap-add and apply a synthesis window
        pin += H  # advance sound pointer
    return y


def sineModelAnal(x, fs, w, N, H, t, maxnSines=100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
    """
    Analysis of a sound using the sinusoidal model with sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, H: hop-size, t: threshold in negative dB
    maxnSines: maximum number of sines per frame, minSineDur: minimum duration of sines in seconds
    freqDevOffset: minimum frequency deviation at 0Hz, freqDevSlope: slope increase of minimum frequency deviation
    returns xtfreq, xtmag, xtphase: frequencies, magnitudes (in dB) and phases of sinusoidal tracks
    """

    if (minSineDur < 0):  # raise error if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1 = int(math.floor((w.size + 1) // 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size // 2))  # half analysis window size by floor
    print('size of x before appending: ', x.size)
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))  # add zeros at the end to analyze last sample
    print('size of x after appending: ', x.size)
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    print('hM1, hM2, pin, pend, H: ', [hM1, hM2, pin, pend, H])
    w = w / sum(w)  # normalize analysis window
    tfreq = np.array([])
    while pin <= pend + 1:  # while input sound pointer is within sound - NB changed < to <= and added 1
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect location indices of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        # perform sinusoidal tracking by adding peaks to trajectories
        tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
        tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
        tmag = np.resize(tmag, min(maxnSines, tmag.size))  # limit number of tracks to maxnSines
        tphase = np.resize(tphase, min(maxnSines, tphase.size))  # limit number of tracks to maxnSines
        jtfreq = np.zeros(maxnSines)  # temporary output array
        jtmag = np.zeros(maxnSines)  # temporary output array
        jtphase = np.zeros(maxnSines)  # temporary output array
        jtfreq[:tfreq.size] = tfreq  # save track frequencies to temporary array
        jtmag[:tmag.size] = tmag  # save track magnitudes to temporary array
        jtphase[:tphase.size] = tphase  # save track phases to temporary array
        if pin == hM1:  # if first frame initialize output sine tracks
            xtfreq = jtfreq
            xtmag = jtmag
            xtphase = jtphase
        else:  # rest of frames append values to sine tracks
            xtfreq = np.vstack((xtfreq, jtfreq))
            xtmag = np.vstack((xtmag, jtmag))
            xtphase = np.vstack((xtphase, jtphase))
        pin += H
    # delete sine tracks shorter than minSineDur
    xtfreq = cleaningSineTracks(xtfreq, round(fs * minSineDur / H))
    return xtfreq, xtmag, xtphase


def sineModelSynth(tfreq, tmag, tphase, M, H, fs):
    """
    Synthesis of a sound using the sinusoidal model
    *** The hop size needs to be the same as what was used for the sineModelAnal ***
    *** (This probably means that the window/FFT size should be about the same as well) ***
    tfreq,tmag,tphase: frequencies, magnitudes (in dB) and phases of sinusoids
    tfreq should be within the range of frequencies captured by an FFT of size M and sampling rate fs
    M: synthesis FFT size (Blackman-Harris window size - should be a power of two for iFFT)
    H: hop size, fs: sampling rate
    returns y: output array sound
    """
    if not isPower2(M):
        M_new = nextbiggestpower2(M)
        warnings.warn("Non-power of 2 FFT size of {} passed to sineModelSynth, using {} instead".format(M, M_new))
        M = M_new

    hM = M // 2  # half analysis window size

    L = tfreq.shape[0]  # number of frames
    pout = 0  # initialize output sound pointer
    ysize = H * (L - 1) + M + 1  # output sound size: H*(L-1) is the number of samples in signal, NB added 1
    print('ysize, M', ysize, M)
    # plus the length of the window
    y = np.zeros(ysize)  # initialize output array
    sw = np.zeros(M)  # initialize synthesis window
    ow = triang(hM)  # triangular window - half of Blackman-Harris window size

    htriM = hM // 2 # half analysis window size
    sw[hM - htriM:hM + htriM] = ow  # add triangular window to the middle of sw

    bh = blackmanharris(M)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hM - htriM:hM + htriM] = sw[hM - htriM:hM + htriM] / bh[hM - htriM:hM + htriM]  # normalized synthesis window
    lastytfreq = tfreq[0, :]  # initialize synthesis frequencies
    ytphase = 2 * np.pi * np.random.rand(tfreq[0, :].size)  # initialize synthesis phases
    for l in range(L):  # iterate over all frames
        if (tphase.size > 0):
            ytphase = tphase[l, :]
        else:   # if no phases generate them
            ytphase += (np.pi * (lastytfreq + tfreq[l, :]) / fs) * H  # propagate phases
        Y = UF.genSpecSines_p(tfreq[l, :], tmag[l, :], ytphase, M, fs)  # generate sines in the spectrum
        lastytfreq = tfreq[l, :]  # save frequency for phase propagation
        ytphase = ytphase % (2 * np.pi)  # make phase inside 2*pi
        yw = np.real(fftshift(ifft(Y)))  # compute inverse FFT
        y[pout:pout + M] += sw * yw  # overlap-add and apply a synthesis window
        pout += H  # advance sound pointer
    y = np.delete(y, range(hM))  # delete half of first window
    y = np.delete(y, range(y.size - hM, y.size))  # delete half of the last window
    return y
