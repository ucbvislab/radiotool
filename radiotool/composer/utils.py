"""A set of utility functions that are used elsewhere in radiotool
"""

import numpy as N

def log_magnitude_spectrum(frames):
    """Compute the log of the magnitude spectrum of frames"""
    return N.log(N.abs(N.fft.rfft(frames)).clip(1e-5, N.inf))


def magnitude_spectrum(frames):
    """Compute the magnitude spectrum of frames"""
    return N.abs(N.fft.rfft(frames))


def RMS_energy(frames):
    """Computes the RMS energy of frames"""
    f = frames.flatten()
    return N.sqrt(N.mean(f * f))

def normalize_features(features):
    """Standardizes features array to fall between 0 and 1"""
    return (features - N.min(features)) / (N.max(features) - N.min(features))

def zero_crossing_last(frames):
    """Finds the last zero crossing in frames"""
    frames = N.array(frames)

    crossings = N.where(N.diff(N.sign(frames)))
    # crossings = N.where(frames[:n] * frames[1:n + 1] < 0)

    if len(crossings[0]) == 0:
        print "No zero crossing"
        return len(frames) - 1
    return crossings[0][-1]


def zero_crossing_first(frames):
    """Finds the first zero crossing in frames"""
    frames = N.array(frames)
    crossings = N.where(N.diff(N.sign(frames)))
    # crossings = N.where(frames[n - 1:-1] * frames[n:] < 0)
    if len(crossings[0]) == 0:
        print "No zero crossing"
        return 0
    return crossings[0][0] + 1

# Crossfading helper methods
# borrowed from echonest remix

def log_factor(arr):
    return N.power(arr, 0.6)


def limiter(arr):
    """
    Restrict the maximum and minimum values of arr
    """
    dyn_range = 32767.0 / 32767.0
    lim_thresh = 30000.0 / 32767.0
    lim_range = dyn_range - lim_thresh

    new_arr = arr.copy()
    
    inds = N.where(arr > lim_thresh)[0]

    new_arr[inds] = (new_arr[inds] - lim_thresh) / lim_range
    new_arr[inds] = (N.arctan(new_arr[inds]) * 2.0 / N.pi) *\
        lim_range + lim_thresh

    inds = N.where(arr < -lim_thresh)[0]

    new_arr[inds] = -(new_arr[inds] + lim_thresh) / lim_range
    new_arr[inds] = -(
        N.arctan(new_arr[inds]) * 2.0 / N.pi * lim_range + lim_thresh)

    return new_arr

def linear(arr1, arr2):
    """
    Create a linear blend of arr1 (fading out) and arr2 (fading in)
    """
    n = N.shape(arr1)[0]
    try: 
        channels = N.shape(arr1)[1]
    except:
        channels = 1
    
    f_in = N.arange(n) / float(n - 1)
    f_out = N.arange(n - 1, -1, -1) / float(n)
    
    if channels > 1:
        f_in = N.tile(f_in, (channels, 1)).T
        f_out = N.tile(f_out, (channels, 1)).T
    
    vals = f_out * arr1 + f_in * arr2
    return vals

def equal_power(arr1, arr2):
    """
    Create an equal power blend of arr1 (fading out) and arr2 (fading in)
    """
    n = N.shape(arr1)[0]
    try: 
        channels = N.shape(arr1)[1]
    except:
        channels = 1
    
    f_in = N.arange(n) / float(n - 1)
    f_out = N.arange(n - 1, -1, -1) / float(n)
    
    if channels > 1:
        f_in = N.tile(f_in, (channels, 1)).T
        f_out = N.tile(f_out, (channels, 1)).T
    
    vals = log_factor(f_out) * arr1 + log_factor(f_in) * arr2

    return limiter(vals)


def segment_array(arr, length, overlap=.5):
    """
    Segment array into chunks of a specified length, with a specified
    proportion overlap.

    Operates on axis 0.

    :param integer length: Length of each segment
    :param float overlap: Proportion overlap of each frame
    """

    arr = N.array(arr)

    offset = float(overlap) * length
    total_segments = int((N.shape(arr)[0] - length) / offset) + 1
    print "total segments", total_segments

    other_shape = N.shape(arr)[1:]
    out_shape = [total_segments, length]
    out_shape.extend(other_shape)

    out = N.empty(out_shape)

    for i in xrange(total_segments):
        out[i][:] = arr[i * offset:i * offset + length]

    return out




