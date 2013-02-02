import numpy as N

from .. import mfcc
from ..composer import Track
from .. import segmentaxis

def mfccs_from_file(filename, swlen=0.250):
    track = Track(filename)
    frames = track.all_as_mono()
    print track.sr()
    return mfccs_from_samples(frames, samplerate=track.sr(), swlen=swlen)


def mfccs_from_samples(frames, samplerate=44100, swlen=0.250):
    subwindow_n_frames = N.power(2, int(N.log2(swlen * samplerate)))
    wlen = subwindow_n_frames / float(samplerate)
    print samplerate
    print "subwindow_n_frames", subwindow_n_frames
    segments = segmentaxis.segment_axis(
        frames, subwindow_n_frames, axis=0,
        overlap=int(subwindow_n_frames * 0.5))
        
    print "segmented axis shape", N.shape(segments)

    m = mfcc.MFCC(samprate=samplerate, wlen=wlen)
    mfccs = N.apply_along_axis(m.frame2s2mfc, 1, segments)
    return mfccs