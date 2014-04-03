import numpy as N
import scipy.signal


class Effect(object):
    """(Abstract) filter for tracks"""
    def __init__(self):
        self.track = track
        self.samplerate = track.samplerate

    def apply_to(self, array):
        return array

    def __str__(self):
        return "Effect"


class NotchFilter(Effect):
    def __init__(self, frequency, gain):
        # http://dsp.stackexchange.com/a/1090
        # right now gain is between 0 and 1. Need to convert this
        # to decibels.
        self.frequency = float(frequency)
        self.gain = float(gain)

    def apply_to(self, array, samplerate):
        # Nyquist frequency
        nyquist = samplerate / 2.

        # ratio of notch freq. to Nyquist freq.
        freq_ratio = self.frequency / nyquist

        # width of the notch
        notchWidth = 0.01

        # Compute zeros
        zeros = N.array([N.exp(1j * N.pi * freq_ratio),
                         N.exp(-1j * N.pi * freq_ratio)])

        # Compute poles
        poles = (1 - notchWidth) * zeros

        b = N.poly(zeros)    # Get moving average filter coefficients
        a = N.poly(poles)    # Get autoregressive filter coefficients

        try:
            if array.ndim == 1:
                filtered = scipy.signal.filtfilt(b, a, array, padtype="even")
            else:
                filtered = N.empty(array.shape)
                for i in range(array.shape[1]):
                    filtered[:, i] = scipy.signal.filtfilt(
                        b, a, array[:, i], padtype="even")

            return filtered * self.gain + array * (1 - self.gain)
        except ValueError:
            print "Could not apply filter"
            return array
