# composer.py
# (c) 2012 - Steve Rubin - srubin@cs.berkeley.edu
# Has all the classes for speech, songs, and fade types
# Additionally, has class for actual composition

import sys

from math import sqrt

import numpy as N

from scikits.audiolab import Sndfile, Format

# import scikits.talkbox as talk
import segmentaxis
import mfcc
from scipy.spatial import distance

# problem on the server with resample...
#from scipy.signal import resample

# import arraypad
from numpy import pad as arraypad

class Composition:
    def __init__(self, tracks=[], channels=2):
        self.tracks = set(tracks)
        self.score = []
        self.dynamics = []
        self.channels = channels

    def add_track(self, track):
        self.tracks.add(track)
    
    def add_tracks(self, tracks):
        self.tracks.update(tracks)
        
    def add_score_segment(self, segment):
        self.score.append(segment)
        
    def add_score_segments(self, segments):
        self.score.extend(segments)

    def add_dynamic(self, dyn):
        self.dynamics.append(dyn)
        
    def add_dynamics(self, dyns):
        self.dynamics.extend(dyns)

    def fade_in(self, segment, duration):
        """adds a fade in (duration in seconds)"""
        dur = int(round(duration * segment.track.samplerate()))
        score_loc_in_seconds = (segment.score_location) /\
            float(segment.track.samplerate())
        f = Fade(segment.track, score_loc_in_seconds, duration, 0.0, 1.0)
        self.add_dynamic(f)
        return f

    def fade_out(self, segment, duration):
        """adds a fade out (duration in seconds)"""
        dur = int(round(duration * segment.track.samplerate()))
        score_loc_in_seconds = (segment.score_location + segment.duration - dur) /\
            float(segment.track.samplerate())
        f = Fade(segment.track, score_loc_in_seconds, duration, 1.0, 0.0)
        self.add_dynamic(f)
        return f

    def extended_fade_in(self, segment, duration):
        """extends the beginning of the segment and adds a fade in
        (duration in seconds)"""
        dur = int(round(duration * segment.track.samplerate()))
        if segment.start - dur >= 0:
            segment.start -= dur
        else:
            raise Exception(
                "Cannot create fade-in that extends past the track's beginning")
        if segment.score_location - dur >= 0:
            segment.score_location -= dur
        else:
            raise Exception(
                "Cannot create fade-in the extends past the score's beginning")

        segment.duration += dur
        
        score_loc_in_seconds = (segment.score_location) /\
            float(segment.track.samplerate())

        f = Fade(segment.track, score_loc_in_seconds, duration, 0.0, 1.0)
        self.add_dynamic(f)
        return f

    def extended_fade_out(self, segment, duration):
        """extends the end of the segment and adds a fade out
        (duration in seconds)"""
        dur = int(round(duration * segment.track.samplerate()))
        if segment.start + segment.duration + dur <\
            segment.track.total_frames():
            segment.duration += dur
        else:
            raise Exception(
                "Cannot create fade-out that extends past the track's end")
        score_loc_in_seconds = (segment.score_location + segment.duration - dur) /\
            float(segment.track.samplerate())
        f = Fade(segment.track, score_loc_in_seconds, duration, 1.0, 0.0)
        self.add_dynamic(f)
        return f
    
    def cross_fade(self, seg1, seg2, duration):
        """equal power crossfade"""
        if seg1.score_location + seg1.duration - seg2.score_location < 2:
            dur = int(duration * seg1.track.samplerate())

            if dur % 2 == 1:
                dur -= 1

            if dur / 2 > seg1.duration:
                dur = seg1.duration * 2

            if dur / 2 > seg2.duration:
                dur = seg2.duration * 2

            # we're going to compute the crossfade and then create a RawTrack
            # for the resulting frames

            seg1.duration += (dur / 2)
            out_frames = seg1.get_frames(channels=self.channels)[-dur:]
            seg1.duration -= dur
            
            seg2.start -= (dur / 2)
            seg2.duration += (dur / 2)
            seg2.score_location -= (dur / 2)
            in_frames = seg2.get_frames(channels=self.channels)[:dur]
            seg2.start += dur
            seg2.duration -= dur
            seg2.score_location += dur

            # compute the crossfade
            in_frames = in_frames[:min(map(len, [in_frames, out_frames]))]
            out_frames = out_frames[:min(map(len, [in_frames, out_frames]))]
            
            cf_frames = equal_power(out_frames, in_frames)
            
            #print "Computed cf_frames", cf_frames
            
            raw_track = RawTrack(cf_frames, name="crossfade",
                samplerate=seg1.track.samplerate())
            
            rs_score_location = (seg1.score_location + seg1.duration) /\
                float(seg1.track.samplerate())
                
            rs_duration = raw_track.duration()
            
            raw_seg = Segment(raw_track, rs_score_location, 0.0, rs_duration)
            
            self.add_track(raw_track)
            self.add_score_segment(raw_seg)
            
            return raw_seg
            
        else:
            print seg1.score_location + seg1.duration, seg2.score_location
            raise Exception("Segments must be adjacent to add a crossfade (%d, %d)" 
                % (seg1.score_location + seg1.duration, seg2.score_location))

    def cross_fade_linear(self, seg1, seg2, duration):
        if seg1.score_location + seg1.duration - seg2.score_location < 2:
            self.extended_fade_out(seg1, duration)
            self.fade_in(seg2, duration)
            # self.extended_fade_in(seg2, duration)
        else:
            print seg1.score_location + seg1.duration, seg2.score_location
            raise Exception("Segments must be adjacent to add a crossfade (%d, %d)"
                % (seg1.score_location + seg1.duration, seg2.score_location))

    def add_music_cue(self, track, score_cue, song_cue, duration=6.0,
                      padding_before=12.0, padding_after=12.0):
        self.tracks.add(track)
        
        pre_fade = 3
        post_fade = 3
        
        if padding_before + pre_fade > song_cue:
            padding_before = song_cue - pre_fade
            
        if padding_before + pre_fade > score_cue:
            padding_before = score_cue - pre_fade
                 
        # print "Composing %s at %.2f from %.2f to %.2f to %.2f to %.2f" % (
        #         track.filename, song_cue, score_cue-padding_before-pre_fade,
        #         score_cue, score_cue+duration,
        #         score_cue+duration+padding_after+post_fade)
        s = Segment(track, score_cue - padding_before - pre_fade,
                    song_cue - padding_before - pre_fade,
                    pre_fade + padding_before + duration + padding_after + post_fade)

        self.add_score_segment(s)
        
        d = []

        dyn_adj = 1
        
        track.set_frame(0)
        
         ## UNCOMMENT THIS STUFF! IT'S CORRECT!
        d.append(Fade(track, score_cue - padding_before - pre_fade, pre_fade,
                      0, .1*dyn_adj, fade_type="linear"))
        d.append(Fade(track, score_cue - padding_before, padding_before,
                      .1*dyn_adj, .4*dyn_adj, fade_type="exponential"))
        d.append(Volume(track, score_cue, duration, .4*dyn_adj))
        d.append(Fade(track, score_cue + duration, padding_after,
                      .4*dyn_adj, 0, fade_type="exponential"))
        # print "\n\n\n\n#####", score_cue+duration+padding_after, post_fade
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

        if long_list[-1] + 16 > len(volumes):
            return frames[:new_cut_point]
        return frames
    
    def build_score(self, **kwargs):
        track_list = kwargs.pop('track', self.tracks)
        adjust_dynamics = kwargs.pop('adjust_dynamics', True)
        min_length = kwargs.pop('min_length', None)

        parts = {}
        starts = {}
        
        # for universal volume adjustment
        all_frames = N.array([])
        song_frames = N.array([])
        speech_frames = N.array([])
        
        longest_part = max([x.score_location + x.duration for x in self.score])
        
        for track_idx, track in enumerate(track_list):
            segments = sorted([v for v in self.score if v.track == track], 
                              key=lambda k: k.score_location + k.duration)
            if len(segments) > 0:
                start_loc = min([x.score_location for x in segments])
                end_loc = max([x.score_location + x.duration for x in segments])
                # end_loc = segments[-1].score_location + segments[-1].duration
                
                starts[track] = start_loc
                
                # print "start loc", start_loc, "end loc", end_loc
                # print "durs", [x.duration for x in segments]

                parts[track] = N.zeros((end_loc - start_loc, self.channels))
                
                for s in segments:
                    frames = s.get_frames(channels=self.channels).\
                        reshape(-1, self.channels)
                    
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
                                
                    parts[track][s.score_location - start_loc:
                                 s.score_location - start_loc + s.duration,
                                 :] = frames

            dyns = sorted([d for d in self.dynamics if d.track == track],
                           key=lambda k: k.score_location)
            for d in dyns:
                vol_frames = d.to_array(self.channels)
                parts[track][d.score_location - start_loc :
                             d.score_location - start_loc + d.duration,
                             :] *= vol_frames

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

        if longest_part < min_length:
            longest_part = min_length
        out = N.zeros((longest_part, self.channels))
        for track, part in parts.iteritems():
            out[starts[track]:starts[track] + len(part)] += part
        
        return out
    
    def output_score(self, **kwargs):
        # get optional args
        filename = kwargs.pop('filename', 'out')
        filetype = kwargs.pop('filetype', 'wav')
        adjust_dynamics = kwargs.pop('adjust_dynamics', True)
        samplerate = kwargs.pop('samplerate', 44100)
        channels = kwargs.pop('channels', 2)
        separate_tracks = kwargs.pop('separate_tracks', False)
        min_length = kwargs.pop('min_length', None)
        
        encoding = 'pcm16'
        if filetype == 'ogg':
            encoding = 'vorbis'
        
        if separate_tracks:
            for track in self.tracks:
                out = self.build_score(track=[track],
                                       adjust_dynamics=adjust_dynamics,
                                       min_length=min_length)
                out_file = Sndfile(filename +"-" + track.name + "." +
                                   filetype, 'w',
                                   Format(filetype, encoding=encoding),
                                   channels, samplerate)
                out_file.write_frames(out)
                out_file.close()

        # always build the complete score
        out = self.build_score(adjust_dynamics=adjust_dynamics,
                               min_length=min_length)

        out_file = Sndfile(filename + "." + filetype, 'w',
                           Format(filetype, encoding=encoding), 
                           channels, samplerate)
        out_file.write_frames(out)
        out_file.close()
        return out
