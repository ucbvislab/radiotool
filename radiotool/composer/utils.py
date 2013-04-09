import numpy as N

def log_magnitude_spectrum(window):
    return N.log(N.abs(N.fft.rfft(window)).clip(1e-5, N.inf))


def magnitude_spectrum(window):
    return N.abs(N.fft.rfft(window))


def RMS_energy(frames):
    f = frames.flatten()
    return N.sqrt(N.mean(f * f))

def normalize_features(features):
    return (features - N.min(features)) / (N.max(features) - N.min(features))


def zero_crossing_last(frames):
    """finds the first zero crossing in frames before frame n"""
    frames = N.array(frames)

    crossings = N.where(N.diff(N.sign(frames)))
    # crossings = N.where(frames[:n] * frames[1:n + 1] < 0)

    if len(crossings[0]) == 0:
        print "No zero crossing"
        return len(frames) - 1
    return crossings[0][-1]


def zero_crossing_first(frames):
    """finds the first zero crossing in frames after frame n"""
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