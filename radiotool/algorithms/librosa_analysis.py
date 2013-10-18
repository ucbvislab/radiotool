
# from librosa examples, and modified by Steve Rubin - srubin@cs.berkeley.edu

import numpy as N
import scipy, scipy.signal
import librosa

HOP = 128
SR  = 44100

def structure2(X):
    print "Computing structure"
    d, n = X.shape

    print "structure shape", X.shape

    X = scipy.stats.zscore(X, axis=1)

    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X.T, metric="cosine"))

    return D[:-1, :-1]

def structure(X, k=3):
    print "Computing structure"
    d, n = X.shape

    X = scipy.stats.zscore(X, axis=1)

    # build the segment-level self-similarity matrix
    D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X.T, metric='cosine'))

    # get the k nearest neighbors of each point
    links       = N.argsort(D, axis=1)[:,1:k+1]

    # get the node clustering
    segments    = N.array(librosa.beat.segment(X, n / 32))

    return links, segments

def analyze_file(infile, debug=True):
    y, sr = librosa.load(infile, sr=SR)
    return analyze_frames(y, sr, debug)

def analyze_frames(y, sr, debug=True):
    A = {}
    
    # First, get the track duration
    A['duration'] = float(len(y)) / sr

    # Then, get the beats
    if debug: print "> beat tracking"
    tempo, beats = librosa.beat.beat_track(y, sr, hop_length=HOP)

    # Push the last frame as a phantom beat
    A['tempo'] = tempo
    A['beats'] = librosa.frames_to_time(beats, sr, hop_length=HOP).tolist()

    print "beats count: ", len(A['beats'])

    # if debug: print "> spectrogram"
    S = librosa.feature.melspectrogram(y, sr,   n_fft=2048, 
                                                hop_length=HOP, 
                                                n_mels=80, 
                                                fmax=8000)
    S = S / S.max()

    # A['spectrogram'] = librosa.logamplitude(librosa.feature.sync(S, beats)**2).T.tolist()

    # Let's make some beat-synchronous mfccs
    if debug: print "> mfcc"
    S = librosa.feature.mfcc(librosa.logamplitude(S), d=40)
    A['timbres'] = librosa.feature.sync(S, beats).T.tolist()

    print "timbres count: ", len(A['timbres'])

    # And some chroma
    if debug: print "> chroma"
    S = N.abs(librosa.stft(y, hop_length=HOP))

    # Grab the harmonic component
    H = librosa.hpss.hpss_median(S, win_P=31, win_H=31, p=1.0)[0]
    A['chroma'] = librosa.feature.sync(librosa.feature.chromagram(H, sr),
                                        beats,
                                        aggregate=N.median).T.tolist()

    # Harmonicity: ratio of H::S averaged per frame
    # if debug: print "> harmonicity"
    # A['harmonicity'] = librosa.feature.sync(N.mean(H / (S + (S==0)), axis=0, keepdims=True),
    #                                         beats,
    #                                         aggregate=N.max).flatten().tolist()


    # Relative loudness
    S = S / S.max()
    S = S**2

    # if debug: print "> loudness"
    # A['loudness'] = librosa.feature.sync(N.max(librosa.logamplitude(S), 
    #                                             axis=0,
    #                                             keepdims=True), 
    #                                      beats, aggregate=N.max).flatten().tolist()

    # Subsample the signal for vis purposes
    # if debug: print "> signal"
    # A['signal'] = scipy.signal.decimate(y, len(y) / 1024, ftype='fir').tolist()

    if debug: print "> dists"
    dists = structure2(N.vstack([N.array(A['timbres']).T, N.array(A['chroma']).T]))
    A['dense_dist'] = dists.tolist()

    edge_lens = [A["beats"][i] - A["beats"][i - 1]
                 for i in xrange(1, len(A["beats"]))]
    A["avg_beat_duration"] = N.mean(edge_lens)

    return A