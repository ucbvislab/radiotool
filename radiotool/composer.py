# composer.py
# (c) 2012 - Steve Rubin - srubin@cs.berkeley.edu
# Has all the classes for speech, songs, and fade types
# Additionally, has class for actual composition

from math import sqrt

import numpy as N
from scikits.audiolab import Sndfile, Format
# import scikits.talkbox as talk
import segmentaxis
import mfcc
from scipy.spatial import distance
# import arraypad
from numpy import pad as arraypad

### Uncomment for MATLAB
# from mlabwrap import mlab as matlab

### note - part of mfcc:
# m = mfcc.MFCC(samprate=s.sr(), wlen=0.1)
# b = m.frame2logspec(f.reshape(2,-1)[1]) clip??

LOG_TO_DB = False
DEBUG = False


if LOG_TO_DB:
    import MySQLdb


def log_magnitude_spectrum(window):
    return N.log(N.abs(N.fft.rfft(window)).clip(1e-5, N.inf))


def magnitude_spectrum(window):
    return N.abs(N.fft.rfft(window))


def RMS_energy(frames):
    f = frames.flatten()
    return N.sqrt(N.mean(f * f))


def IS_distance(p1, p2):
    """Calculate the Itakura-Saito spectral distance between power spectra"""
    # see 
    # http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/distispf.html
   
    # but implementing this... 
    # http://hil.t.u-tokyo.ac.jp/~kameoka/SAP/papers/El-Jaroudi1991__Discrete-All_Pole_Modeling.pdf
    # equation 14
    
    if len(N.where(p2 == 0)[0]) > 0: return 0
    q = p1 / p2
    if N.isinf(N.mean(q - N.log(q))): return 0
    return N.mean(q - N.log(q)) - 1
    
def COSH_distance(p1, p2):
    """IS distance is asymmetric, so this accounts for that"""
    return (IS_distance(p1, p2) + IS_distance(p2, p1)) / 2

def robust_logistic_regression(features):
    mu = N.mean(features)
    sigma = N.std(features)
    gamma = N.log(99) # natural log
    return 1 / (1 + N.exp(-gamma*(features-mu)/sigma))
    
def normalize_features(features):
    return (features - N.min(features)) / (N.max(features) - N.min(features))

class Track:
    # TODO: add in mp3 compatibility (convert to wav if necessary)
    def __init__(self, fn, name="No name"):
        """Create a Track object from a music filename"""
        self.filename = fn
        self.name = name
        try:
            self.sound = Sndfile(self.filename, 'r')
            self.current_frame = 0
            self.channels = self.sound.channels
        except:
            print 'Could not open track: %s' % self.filename

    def read_frames(self, n):
        if self.channels == 1:
            out = N.zeros(n)
        elif self.channels == 2:
            out = N.zeros((n,2))
        else:
            print "Input needs to have 1 or 2 channels"
            return
        if n > self.remaining_frames():
            print "Trying to retrieve too many frames!"
            n = self.remaining_frames()
        self.current_frame += n
        out[:n, :] = self.sound.read_frames(n)
        return out
        
    def set_frame(self, n):
        self.sound.seek(n)
        self.current_frame = n
    
    def reset(self):
        self.set_frame(0)
        self.current_frame = 0

    def all_as_mono(self):
        """Get the entire track as 1 combined channel"""
        tmp_current = self.current_frame
        self.reset()
        tmp_frames = self.read_frames(self.total_frames())
        if self.channels == 2:
            frames = tmp_frames[:, 0].copy() + tmp_frames[:, 1].copy()
        elif self.channels == 1:
            frames = tmp_frames
        else:
            raise IOError("Input audio must have either 1 or 2 channels")
        self.current_frame = tmp_current
        return frames

    def samplerate(self):
        return self.sound.samplerate
        
    def sr(self):
        return self.samplerate()

    def remaining_frames(self):
        return self.sound.nframes - self.current_frame
        
    def total_frames(self):
        return self.sound.nframes
        
    def loudest_time(self, start=0, duration=0):
        """Find the loudest time in the window given by start and duration
        Returns frame number in context of entire track, not just the window
        """
        if duration == 0:
            duration = self.sound.nframes
        self.set_frame(start)
        arr = self.read_frames(duration)
        # get the frame of the maximum amplitude
        # different names for the same thing...
        # max_amp_sample = a.argmax(axis=0)[a.max(axis=0).argmax()]
        max_amp_sample = int(N.floor(arr.argmax()/2)) + start
        return max_amp_sample
    
    def refine_cut(self, cut_point, window_size=1):
        return cut_point
    
class Song(Track):
    def __init__(self, fn, name="Song name"):
        Track.__init__(self, fn, name)
        
    def magnitude_spectrum(self, window):
        """Compute the magnitude spectra"""
        return N.abs(N.fft.rfft(window))
        
    def partial_mfcc(self, window):
        """partial mfcc calculation (stopping before mel band filter)"""
  
        dump_out["names"] = ('MFCC euclidean distance',
                             'RMS energy distance',
                             'Chromagram COSH distance',
                             'Chromagram euclidean distance',
                             'Tempo difference',
                             'Magnitude spectra COSH distance',
                             'RMS energy')
   
    def refine_cut_by(self, refinement, cut_point, window_size=4):
        if refinement == "RMS energy distance":
            return self.refine_cut_rms_jump(cut_point, window_size)
        elif refinement == "MFCC euclidean distance":
            return self.refine_cut_mfcc_euc(cut_point, window_size)
        elif refinement == "Chromagram euclidean distance":
            return self.refine_cut_chroma_euc(cut_point, window_size)
            
        return self.refine_cut_rms_jump(cut_point, window_size)
    
    def refine_cut_rms_jump(self, cut_point, window_size=4):
        # subwindow length
        swlen = 0.250 # 250ms 
        
        start_frame = int((cut_point - window_size * 0.5) * self.sr())
        if (start_frame < 0):
            start_frame = 0
        
        if (start_frame + window_size * self.sr() > self.total_frames()):
            start_frame = self.total_frames() - window_size * self.sr() - 1
            
        self.set_frame(start_frame)
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
        overlap=int(subwindow_n_frames * 0.5))  
        
        RMS_energies = N.apply_along_axis(RMS_energy, 1, segments) 
           
        energy_diffs = N.zeros(len(RMS_energies))
        energy_diffs[1:] = RMS_energies[1:] - RMS_energies[:-1]
        idx = N.where(energy_diffs == max(energy_diffs))[0][0]
        return round(cut_point - window_size * 0.5 +
                           idx * swlen * 0.5, 2), \
               normalize_features(energy_diffs)

    def refine_cut_mfcc_euc(self, cut_point, window_size=4):
        return self.refine_cut_mfcc(cut_point, window_size, "euclidean")
    
    def refine_cut_mfcc(self, cut_point, window_size=4, dist="euclidean"):
        # subwindow length
        swlen = 0.250 #  
        
        self.set_frame(int((cut_point - window_size * 0.5) * self.sr()))
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
        overlap=int(subwindow_n_frames * 0.5))
        # compute MFCCs, compare Euclidean distance
        m = mfcc.MFCC(samprate=self.sr(), wlen=swlen)
        mfccs = N.apply_along_axis(m.frame2s2mfc, 1, segments)
        mfcc_dists = N.zeros(len(mfccs))
        for i in range(1,len(mfcc_dists)):
            if dist == "euclidean":
                mfcc_dists[i] = N.linalg.norm(mfccs[i-1] - mfccs[i])
            elif dist == "cosine":
                mfcc_dists[i] = distance.cosine(mfccs[i-1], mfccs[i])
        if DEBUG: print "MFCC euclidean distances: ", mfcc_dists
        idx = N.where(mfcc_dists == max(mfcc_dists))[0][0]
        return round(cut_point - window_size * 0.5 +
                           idx * swlen * 0.5, 2), \
               normalize_features(mfcc_dists)
                           
    def refine_cut_chroma_euc(self, cut_point, window_size=4):
        # subwindow length
        swlen = 0.24 #  
        
        self.set_frame(int((cut_point - window_size * 0.5) * self.sr()))
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
        overlap=int(subwindow_n_frames * 0.5))
        # compute chromagram
        fftlength = 44100 * swlen
        # this compute with 3/4 overlapping windows and we want
        # 1/2 overlapping, so we'll take every other column
        cgram = matlab.chromagram_IF(frames, 44100, fftlength)
        # don't need to get rid of 3/4 overlap because we're using it
        # on its own
        # cgram_idx = range(0, len(cgram[0,:]), 2)
        # cgram = cgram[:,cgram_idx]
        cgram_euclidean = N.array([N.linalg.norm(cgram[:,i] - cgram[:,i+1])
                                  for i in range(len(cgram[0,:])-1)])
        idx = N.where(cgram_euclidean == max(cgram_euclidean))[0][0]
        return round(cut_point - window_size * 0.5 +
                     (idx + 1) * swlen * .25, 2), ()
        
    def refine_cut(self, cut_point, window_size=2, scored=True):
        # these should probably all be computed elsewhere and merged
        # (scored?) here
        
        cut_idx = {}
        features = {}
        
        # subwindow length
        swlen = 0.1 # 100ms 
        
        self.set_frame(int((cut_point - window_size * 0.5) * self.sr()))
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
                                     overlap=int(subwindow_n_frames * 0.5))

        # should I not use the combined left+right for this feature?
        RMS_energies = N.apply_along_axis(RMS_energy, 1, segments)
        
        if DEBUG: print "RMS energies: ", RMS_energies
        # this is probably not a great feature
        #features["rms_energy"] = RMS_energies
        cut_idx["rms_energy"] = N.where(RMS_energies == max(RMS_energies))[0][0]
        
        ## do it by biggest jump between windows instead
        ## disregard overlapping windows for now
        energy_diffs = N.zeros(len(RMS_energies))
        energy_diffs[1:] = RMS_energies[1:] - RMS_energies[:-1]
        if DEBUG: print "energy differences: ", energy_diffs
        features["rms_jump"] = energy_diffs
        cut_idx["rms_jump"] = N.where(energy_diffs == max(energy_diffs))[0][0]
        
        # compute power spectra, compare differences with I-S distance
        magnitude_spectra = N.apply_along_axis(self.magnitude_spectrum,
                                               1, segments)
        #IS_ms_distances = N.zeros(len(magnitude_spectra))
        # is there a better way... list comprehensions with numpy?
        # for i in range(1,len(IS_ms_distances)):
        #     # not symmetric... do average?
        #     IS_ms_distances[i] = COSH_distance(magnitude_spectra[i-1],
        #                                   magnitude_spectra[i])
        IS_ms_distances = N.array([
            COSH_distance(magnitude_spectra[i],
                          magnitude_spectra[i+1])
            for i in range(len(magnitude_spectra)-1)])
        IS_ms_distances = N.append(IS_ms_distances, 0)
        
        if DEBUG: print "IS ms distances", IS_ms_distances
        features["magnitude_spectra_COSH"] = IS_ms_distances
        cut_idx["magnitude_spectra_COSH"] = N.where(
                IS_ms_distances == max(IS_ms_distances))[0][0] + 1
                
        # compute MFCCs, compare Euclidean distance
        m = mfcc.MFCC(samprate=self.sr(), wlen=swlen)
        mfccs = N.apply_along_axis(m.frame2s2mfc, 1, segments)
        mfcc_dists = N.zeros(len(mfccs))
        for i in range(1,len(mfcc_dists)):
            mfcc_dists[i] = N.linalg.norm(mfccs[i-1] - mfccs[i])
        if DEBUG: print "MFCC euclidean distances: ", mfcc_dists
        features["mfcc_euclidean"] = mfcc_dists
        cut_idx["mfcc_euclidean"] = N.where(mfcc_dists ==
                                            max(mfcc_dists))[0][0]
        
         
        
        combined_features = N.zeros(len(segments))
        for k, v in features.iteritems():
            combined_features += (v - min(v)) / (max(v)- min(v))
        
        cut_idx["combined"] = N.where(combined_features == 
                                      max(combined_features))[0][0]
        if DEBUG: print 'Combined features: ', combined_features
        
        IDX = 'mfcc_euclidean'
        if DEBUG: print "Using ", IDX
                                  
        for k, v in cut_idx.iteritems():
            cut_idx[k] = round(cut_point - window_size * 0.5 +
                               v * swlen * 0.5, 2)
            
        from pprint import pprint            
        if DEBUG: pprint(cut_idx)
        
        # log results to DB for later comparison
        if LOG_TO_DB:
            try:
                con = MySQLdb.connect('localhost', 'root',
                                      'qual-cipe-whak', 'music')
                cur = con.cursor(MySQLdb.cursors.DictCursor)
                desc = "Highest MFCC euclidean distance " + \
                       "with 4 second window, .1 second subwindow and " + \
                       "euclidean distance MFCC segmentation (4 second window)"
                method_q = "SELECT * FROM methods WHERE description = '%s'" \
                            % desc
                cur.execute(method_q)
                method = cur.fetchone()
            
                if method is None:
                    query = "INSERT INTO methods(description) VALUES('%s')" \
                            % desc
                    cur.execute(query)
                    cur.execute(method_q)
                    method = cur.fetchone()
                
                method_id = method["id"]
            
                fn = '.'.join(self.filename.split('/')[-1]
                              .split('.')[:-1]) + '%'
                song_q = "SELECT * FROM songs WHERE filename LIKE %s"
                cur.execute(song_q, fn)
                song = cur.fetchone()
            
                if song is None:
                    print "Could not find song in db matching filename %s" % (
                        filename)
                    return cut_idx[IDX]

                song_id = song["id"]

                result_q = "INSERT INTO results(song_id, song_cutpoint, " + \
                           "method_id) VALUES(%d, %f, %d)" % (song_id, 
                           cut_idx[IDX], method_id)
                cur.execute(result_q)
            
            except MySQLdb.Error, e:
                print "Error %d: %s" % (e.args[0], e.args[1])

            finally:
                if cur:
                    cur.close()
                if con:
                    con.commit()
                    con.close()
        
        return cut_idx[IDX]
    
class Speech(Track):
    def __init__(self, fn, name="Speech name"):
        Track.__init__(self, fn, name)
    
    def refine_cut(self, cut_point, window_size=1):
        self.set_frame(int((cut_point - window_size / 2.0) * self.sr()))
        frames = self.read_frames(window_size * self.sr())
        subwindow_n_frames = int((window_size / 16.0) * self.sr())

        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
                                     overlap=int(subwindow_n_frames / 2.0))

        segments = segments.reshape((-1, subwindow_n_frames * 2))
        #volumes = N.mean(N.abs(segments), 1)
        volumes = N.apply_along_axis(RMS_energy, 1, segments)
 
        if DEBUG: print volumes
        min_subwindow_vol = min(N.sum(N.abs(segments), 1) / subwindow_n_frames)
        min_subwindow_vol = min(volumes)
        if DEBUG: print min_subwindow_vol
        # some threshold? what if there are no zeros?
        
        min_subwindow_vol_index = N.where(volumes <= 1.1 * 
                                          min_subwindow_vol)

        # first_min_subwindow = min_subwindow_vol_index[0][0]
        # closest_min_subwindow = find_nearest(min_subwindow_vol_index[0], 
        #                                      len(volumes)/2)
        
        # find longest span of "silence" and set to the beginning
        # adapted from 
        # http://stackoverflow.com/questions/3109052/
        # find-longest-span-of-consecutive-array-keys
        last_key = -1
        cur_list = []
        long_list = []
        for idx in min_subwindow_vol_index[0]:
            if idx != last_key + 1:
                cur_list = []
            cur_list.append(idx)
            if(len(cur_list) > len(long_list)):
                long_list = cur_list
            last_key = idx
        
        new_cut_point = (self.sr() * (cut_point - window_size / 2.0) + 
                         (long_list[0] + 1) * 
                         int(subwindow_n_frames / 2.0))
        print "first min subwindow", long_list[0], "total", len(volumes)
        return round(new_cut_point / self.sr(), 2)
        # have to add the .5 elsewhere to get that effect!
        
class Segment:
    # score_location, start, and duration all in seconds
    # -- may have to change later if this isn't accurate enough
    def __init__(self, track, score_location, start, duration):
        self.samplerate = track.samplerate()
        self.track = track
        self.score_location = int(score_location * self.samplerate)
        self.start = int(start * self.samplerate)
        self.duration = int(duration * self.samplerate)
        
class Dynamic:
    def __init__(self, track, score_location, duration):
        self.track = track
        self.samplerate = track.samplerate()
        self.score_location = int(round(score_location * self.samplerate))
        self.duration = int(round(duration * self.samplerate))
        
    def to_array(self):
        return N.ones( (self.duration, 2) )
        
    def __str__(self):
        return "Dynamic at %d with duration %d" % (self.score_location,
                                                   self.duration)
        
class Volume(Dynamic):
    def __init__(self, track, score_location, duration, volume):
        Dynamic.__init__(self, track, score_location, duration)
        self.volume = volume
        
    def to_array(self):
        return N.linspace(self.volume, self.volume, 
                        self.duration*2).reshape(self.duration, 2)
        
class Fade(Dynamic):
    # linear, exponential, (TODO: cosine)
    def __init__(self, track, score_location, duration, 
                in_volume, out_volume, fade_type="linear"):
        Dynamic.__init__(self, track, score_location, duration)
        self.in_volume = in_volume
        self.out_volume = out_volume
        self.fade_type = fade_type
        
    def to_array(self):
        if self.fade_type == "linear":
            return N.linspace(self.in_volume, self.out_volume, 
                            self.duration*2).reshape(self.duration, 2)
        elif self.fade_type == "exponential":
            if self.in_volume < self.out_volume:
                return (N.logspace(8, 1, self.duration*2, base=.5) * (
                                self.out_volume - self.in_volume) / 0.5 + 
                                self.in_volume).reshape(self.duration, 2)
            else:
                return (N.logspace(1, 8, self.duration*2, base=.5) * (
                                self.in_volume - self.out_volume) / 0.5 + 
                                self.out_volume).reshape(self.duration, 2)
        elif self.fade_type == "cosine":
            return

class Composition:
    def __init__(self, tracks=[]):
        self.tracks = set(tracks)
        self.score = []
        self.dynamics = []

    def add_track(self, track):
        self.tracks.add(track)
        
    def add_score_segment(self, segment):
        self.score.append(segment)
        
    def add_score_segments(self, segments):
        self.score.extend(segments)

    def add_dynamic(self, dyn):
        self.dynamics.append(dyn)
        
    def add_dynamics(self, dyns):
        self.dynamics.extend(dyns)
    
    def add_music_cue(self, track, score_cue, song_cue, duration=6.0,
                      padding_before=12.0, padding_after=12.0):
        self.tracks.add(track)
        
        pre_fade = 3
        post_fade = 3
        
        if padding_before + pre_fade > song_cue:
            padding_before = song_cue - pre_fade
            
        if padding_before + pre_fade > score_cue:
            padding_before = score_cue - pre_fade
                 
        print "Composing %s at %.2f from %.2f to %.2f to %.2f to %.2f" % (
                track.filename, song_cue, score_cue-padding_before-pre_fade,
                score_cue, score_cue+duration,
                score_cue+duration+padding_after+post_fade)
        s = Segment(track, score_cue - padding_before - pre_fade,
                    song_cue - padding_before - pre_fade,
                    pre_fade + padding_before + duration + padding_after + post_fade)

        self.add_score_segment(s)
        
        d = []
        # # FUTURE WORK: learn how to adjust volumes -- normalize gain?
        # # Giving it a shot here (normalizing by RMS energy)
        # track.set_frame(track.sr() * song_cue)
        # energy_window = track.sr() * (pre_fade + padding_before + duration +
        #                               padding_after + post_fade)
        # energy = RMS_energy(track.read_frames(energy_window))
        # #print "Music energy", energy
        # dyn_adj = 0.10 / energy
        # #track.set_frame(0)
        # #track.set_frame(track.sr() * song_cue)
        dyn_adj = 1
        
        #print "Normalized music energy", RMS_energy(track.read_frames(energy_window) * dyn_adj)
        
        track.set_frame(0)
        
         ## UNCOMMENT THIS STUFF! IT'S CORRECT!
        d.append(Fade(track, score_cue - padding_before - pre_fade, pre_fade,
                      0, .1*dyn_adj, fade_type="linear"))
        d.append(Fade(track, score_cue - padding_before, padding_before,
                      .1*dyn_adj, .4*dyn_adj, fade_type="exponential"))
        d.append(Volume(track, score_cue, duration, .4*dyn_adj))
        d.append(Fade(track, score_cue + duration, padding_after,
                      .4*dyn_adj, 0, fade_type="exponential"))
        print "\n\n\n\n#####", score_cue+duration+padding_after, post_fade
        d.append(Fade(track, score_cue + duration + padding_after, post_fade,
                      .1*dyn_adj, 0, fade_type="linear"))
        self.add_dynamics(d)
    
    def _remove_end_silence(self, frames):
        subwindow_n_frames = int(1/16.0 * 44100)

        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
                                     overlap=int(subwindow_n_frames / 2.0))

        # segments = segments.reshape((-1, subwindow_n_frames * 2))
        #volumes = N.mean(N.abs(segments), 1)
        volumes = N.apply_along_axis(RMS_energy, 1, segments)

        if DEBUG: print volumes
        min_subwindow_vol = min(N.sum(N.abs(segments), 1) /\
                            subwindow_n_frames)
        min_subwindow_vol = min(volumes)
        if DEBUG: print min_subwindow_vol
        # some threshold? what if there are no zeros?
    
        min_subwindow_vol_index = N.where(volumes <= 2.0 * 
                                          min_subwindow_vol)

        # first_min_subwindow = min_subwindow_vol_index[0][0]
        # closest_min_subwindow = find_nearest(min_subwindow_vol_index[0], 
        #                                      len(volumes)/2)
    
        # find longest span of "silence" and set to the beginning
        # adapted from 
        # http://stackoverflow.com/questions/3109052/
        # find-longest-span-of-consecutive-array-keys
        last_key = -1
        cur_list = []
        long_list = []
        for idx in min_subwindow_vol_index[0]:
            if idx != last_key + 1:
                cur_list = []
            cur_list.append(idx)
            if(len(cur_list) > len(long_list)):
                long_list = cur_list
            last_key = idx
    
        new_cut_point =  (long_list[0] + 1) * \
                         int(subwindow_n_frames / 2.0)
        # print "first min subwindow", long_list[0], "total", len(volumes)
        # # return round(new_cut_point / self.sr(), 2)
        # print "went from ", len(frames), " to ", new_cut_point
        # print long_list, min_subwindow_vol_index
        # print long_list[-1], len(volumes) - 1
        if long_list[-1] + 16 > len(volumes):
            return frames[:new_cut_point]
        return frames

    
    def build_score(self, **kwargs):
        track_list = kwargs.pop('track', self.tracks)
        adjust_dynamics = kwargs.pop('adjust_dynamics', True)
        
        parts = {}
        longest_part = 0
        
        # for universal volume adjustment
        all_frames = N.array([])
        song_frames = N.array([])
        speech_frames = N.array([])
        
        for track in track_list:
            segments = sorted([v for v in self.score if v.track == track], 
                              key=lambda k: k.score_location)
            if len(segments) > 0:
                parts[track] = N.zeros( (segments[-1].score_location + 
                                         segments[-1].duration, 2) )
                if segments[-1].score_location +\
                   segments[-1].duration > longest_part:
                    longest_part = segments[-1].score_location +\
                                   segments[-1].duration
                for s in segments:
                    # print "### segment ", s, s.track
                    track.set_frame(s.start)
                    frames = track.read_frames(s.duration)
                    
                    # for universal volume adjustment
                    if adjust_dynamics:
                        all_frames = N.append(all_frames,
                            self._remove_end_silence(frames.flatten()))
                        if isinstance(track, Song):
                            song_frames = N.append(song_frames, 
                                self._remove_end_silence(frames.flatten()))
                        elif isinstance(track, Speech):
                            speech_frames = N.append(speech_frames,
                                self._remove_end_silence(frames.flatten()))
                    
                    parts[track].put(N.arange(s.score_location*2, 
                                    ( s.score_location+s.duration )*2), 
                                      frames)
                    # print "last frame of segment is: ",\
                           # s.score_location + s.duration
                    
            dyns = sorted([d for d in self.dynamics if d.track == track],
                           key=lambda k: k.score_location)
            for d in dyns:
                # EXPLAIN -2 addend! Array indexing, basically. Starts at 0.
                dyn_range = N.arange(d.score_location * 2 - 2, 
                                       (d.score_location+d.duration)*2 - 2)
                adjusted = (parts[track].take(dyn_range).\
                           reshape(d.duration, 2) * d.to_array())
                print adjusted
                parts[track].put(N.arange(d.score_location * 2 - 2,
                                     (d.score_location+d.duration) * 2 - 2),
                                      adjusted)
                print "last frame of dynamic is: ",\
                    d.score_location+d.duration
                
                # dyn_range = N.arange(d.score_location * 2, 
                #                      (d.score_location+d.duration)*2)
                # adjusted = (parts[track].take(dyn_range).reshape(d.duration, 2) 
                #                 * d.to_array())
                # parts[track].put(N.arange(d.score_location * 2,
                #                      (d.score_location+d.duration) * 2),
                #                       adjusted)
        
        if adjust_dynamics:
            total_energy = RMS_energy(all_frames)
            song_energy = RMS_energy(song_frames)
            speech_energy = RMS_energy(speech_frames)
                
        # dyn_adj = 0.10 / total_energy
        # dyn_adj = speech_energy / sqrt(song_energy) * 5
        if adjust_dynamics:
            if not N.isnan(speech_energy) and not N.isnan(song_energy):
                dyn_adj = sqrt(speech_energy / song_energy) * 1.15
            else:
                dyn_adj = 1
        else:
            dyn_adj = 1
            
        print "\n\n### Multiplying song signal by ", dyn_adj
        
        out = N.zeros( (longest_part, 2))
        for track, part in parts.iteritems():
            # TODO: -3 second hack -- fix later
            # out[:len(part)-132300] += part[:-132300]
            if isinstance(track, Song):
                print "Dyn adjusting song"
                out[:len(part)-132300] += part[:-132300] * dyn_adj
            else:
                print "Dyn adjusting speech"
                out[:len(part)] += part
        
        return out
    
    def output_score(self, **kwargs):
        # get optional args
        filename = kwargs.pop('filename', 'out')
        filetype = kwargs.pop('filetype', 'wav')
        adjust_dynamics = kwargs.pop('adjust_dynamics', True)
        samplerate = kwargs.pop('samplerate', 44100)
        separate_tracks = kwargs.pop('separate_tracks', False)
        
        if separate_tracks:
            for track in self.tracks:
                out = self.build_score(track=[track],
                                       adjust_dynamics=adjust_dynamics)
                out_file = Sndfile(filename +"-" + track.name + "." +
                                   filetype, 'w', Format(filetype),
                                   2, samplerate)
                out_file.write_frames(out)
                out_file.close()

        # always build the complete score
        out = self.build_score(adjust_dynamics=adjust_dynamics)
        out_file = Sndfile(filename + "." + filetype, 'w',
                           Format(filetype), 2, samplerate)
        out_file.write_frames(out)
        out_file.close()
        return out
        