import sys
from math import sqrt

import numpy as np

from scikits.audiolab import Sndfile, Format
try:
    import libxmp
    LIBXMP = True
except:
    LIBXMP = False

from rawtrack import RawTrack
from fade import Fade
from segment import Segment
from volume import Volume
from ..utils import equal_power, RMS_energy, segment_array, wav_to_mp3
import radiotool.utils


class Composition(object):
    """
    Create a composition made up of bits of different audio
    tracks.
    """

    def __init__(self, tracks=None, channels=2, segments=None, dynamics=None,
                 labels=None):
        """Initialize a composition with optional starting tracks/segments.

        :param tracks: Initial tracks in the composition
        :type tracks: list of :py:class:`radiotool.composer.Track`
        :param channels: Number of channels in the composition
        :type channels: integer
        :param segments: Initial segments in the composition
        :type segments: list of :py:class:`radiotool.composer.Segment`
        :param dynamics: Initial dynamics in the composition
        :type dynamics: list of :py:class:`radiotool.composer.Dynamic`
        :returns: A new composition
        :rtype: Composition

        """
        if tracks is None:
            self.tracks = set()
        else:
            self.tracks = set(tracks)

        if segments is None:
            self.segments = []
        else:
            self.segments = list(segments)

        if dynamics is None:
            self.dynamics = []
        else:
            self.dynamics = list(dynamics)

        if labels is None:
            self.labels = []
        else:
            self.labels = list(labels)

        self.channels = channels

    @property
    def duration(self):
        """Get duration of composition
        """
        return max([x.comp_location + x.duration
                    for x in self.segments])

    def add_track(self, track):
        """Add track to the composition

        :param track: Track to add to composition
        :type track: :py:class:`radiotool.composer.Track`

        """
        self.tracks.add(track)

    def add_tracks(self, tracks):
        """Add a list of tracks to the composition

        :param tracks: Tracks to add to composition
        :type tracks: list of :py:class:`radiotool.composer.Track`
        """
        self.tracks.update(tracks)

    def add_segment(self, segment):
        """Add a segment to the composition

        :param segment: Segment to add to composition
        :type segment: :py:class:`radiotool.composer.Segment`
        """
        self.tracks.add(segment.track)
        self.segments.append(segment)

    def add_segments(self, segments):
        """Add a list of segments to the composition

        :param segments: Segments to add to composition
        :type segments: list of :py:class:`radiotool.composer.Segment`
        """
        self.tracks.update([seg.track for seg in segments])
        self.segments.extend(segments)

    def add_dynamic(self, dyn):
        """Add a dynamic to the composition

        :param dyn: Dynamic to add to composition
        :type dyn: :py:class:`radiotool.composer.Dynamic`
        """
        self.dynamics.append(dyn)

    def add_dynamics(self, dyns):
        """Add a list of dynamics to the composition

        :param dyns: Dynamics to add to composition
        :type dyns: list of :py:class:`radiotool.composer.Dynamic`
        """
        self.dynamics.extend(dyns)

    def add_label(self, label):
        """Add a label to the composition

        :param label: Label to add to composition
        :type label: :py:class:`radiotool.composer.Label`
        """
        self.labels.append(label)

    def add_labels(self, labels):
        """Add a list of labels to the composition

        :param labels: List of labels to add to composition
        :type label: list of :py:class:`radiotool.composer.Label`
        """
        self.labels.extend(labels)

    def fade_in(self, segment, duration, fade_type="linear"):
        """Adds a fade in to a segment in the composition

        :param segment: Segment to fade in to
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-in (in seconds)
        :type duration: float
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        f = Fade(segment.track, segment.comp_location_in_seconds,
                 duration, 0.0, 1.0, fade_type=fade_type)
        self.add_dynamic(f)
        return f

    def fade_out(self, segment, duration, fade_type="linear"):
        """Adds a fade out to a segment in the composition

        :param segment: Segment to fade out
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-out (in seconds)
        :type duration: float
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        score_loc_in_seconds = segment.comp_location_in_seconds +\
            segment.duration_in_seconds - duration

        f = Fade(segment.track, score_loc_in_seconds, duration, 1.0, 0.0,
                 fade_type=fade_type)
        # bug fixing... perhaps
        f.comp_location = segment.comp_location + segment.duration -\
            int(duration * segment.track.samplerate)
        self.add_dynamic(f)
        return f

    def extended_fade_in(self, segment, duration):
        """Add a fade-in to a segment that extends the beginning of the
        segment.

        :param segment: Segment to fade in
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-in (in seconds)
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        dur = int(duration * segment.track.samplerate)
        if segment.start - dur >= 0:
            segment.start -= dur
        else:
            raise Exception(
                "Cannot create fade-in that extends "
                "past the track's beginning")
        if segment.comp_location - dur >= 0:
            segment.comp_location -= dur
        else:
            raise Exception(
                "Cannot create fade-in the extends past the score's beginning")

        segment.duration += dur

        f = Fade(segment.track, segment.comp_location_in_seconds,
                 duration, 0.0, 1.0)
        self.add_dynamic(f)
        return f

    def extended_fade_out(self, segment, duration):
        """Add a fade-out to a segment that extends the beginning of the
        segment.

        :param segment: Segment to fade out
        :type segment: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of fade-out (in seconds)
        :returns: The fade that has been added to the composition
        :rtype: :py:class:`Fade`
        """
        dur = int(duration * segment.track.samplerate)
        if segment.start + segment.duration + dur <\
                segment.track.duration:
            segment.duration += dur
        else:
            raise Exception(
                "Cannot create fade-out that extends past the track's end")
        score_loc_in_seconds = segment.comp_location_in_seconds +\
            segment.duration_in_seconds - duration
        f = Fade(segment.track, score_loc_in_seconds, duration, 1.0, 0.0)
        self.add_dynamic(f)
        return f

    def cross_fade(self, seg1, seg2, duration):
        """Add a linear crossfade to the composition between two
        segments.

        :param seg1: First segment (fading out)
        :type seg1: :py:class:`radiotool.composer.Segment`
        :param seg2: Second segment (fading in)
        :type seg2: :py:class:`radiotool.composer.Segment`
        :param duration: Duration of crossfade (in seconds)
        """

        if seg1.comp_location + seg1.duration - seg2.comp_location < 2:
            dur = int(duration * seg1.track.samplerate)

            if dur % 2 == 1:
                dur -= 1

            if dur / 2 > seg1.duration:
                dur = seg1.duration * 2

            if dur / 2 > seg2.duration:
                dur = seg2.duration * 2

            # we're going to compute the crossfade and then create a RawTrack
            # for the resulting frames

            if seg2.start - (dur / 2) < 0:
                diff = seg2.start
                seg2.start = 0
                seg2.duration -= diff
                seg2.comp_location -= diff
                dur = 2 * diff
            else:
                seg2.start -= (dur / 2)
                seg2.duration += (dur / 2)
                seg2.comp_location -= (dur / 2)

            seg1.duration += (dur / 2)
            out_frames = seg1.get_frames(channels=self.channels)[-dur:]
            seg1.duration -= dur

            in_frames = seg2.get_frames(channels=self.channels)[:dur]
            seg2.start += dur
            seg2.duration -= dur
            seg2.comp_location += dur

            # compute the crossfade
            in_frames = in_frames[:min(map(len, [in_frames, out_frames]))]
            out_frames = out_frames[:min(map(len, [in_frames, out_frames]))]

            cf_frames = radiotool.utils.linear(out_frames, in_frames)
            # cf_frames = equal_power(out_frames, in_frames)

            raw_track = RawTrack(cf_frames, name="crossfade",
                                 samplerate=seg1.track.samplerate)

            rs_comp_location = (seg1.comp_location + seg1.duration) /\
                float(seg1.track.samplerate)

            rs_duration = raw_track.duration / float(raw_track.samplerate)

            raw_seg = Segment(raw_track, rs_comp_location, 0.0, rs_duration)
            # will this fix a bug?
            raw_seg.duration = raw_track.duration
            raw_seg.comp_location = seg1.comp_location + seg1.duration

            self.add_track(raw_track)
            self.add_segment(raw_seg)

            return raw_seg

        else:
            print seg1.comp_location + seg1.duration, seg2.comp_location
            raise Exception("Segments must be adjacent"
                            "to add a crossfade ({}, {})".format(
                            seg1.comp_location + seg1.duration,
                            seg2.comp_location))

    def cross_fade_linear(self, seg1, seg2, duration):
        if seg1.comp_location + seg1.duration - seg2.comp_location < 2:
            self.extended_fade_out(seg1, duration)
            self.fade_in(seg2, duration)
            # self.extended_fade_in(seg2, duration)
        else:
            print seg1.comp_location + seg1.duration, seg2.comp_location
            raise Exception("Segments must be adjacent to add a crossfade (%d, %d)"
                % (seg1.comp_location + seg1.duration, seg2.comp_location))

    def empty_over_span(self, time, duration):
        """Helper method that tests whether composition contains any segments
        at a given time for a given duration. 

        :param time: Time (in seconds) to start span
        :param duration: Duration (in seconds) of span
        :returns: `True` if there are no segments in the composition that overlap the span starting at `time` and lasting for `duration` seconds. `False` otherwise.
        """
        for seg in self.segments:
            # starts in range
            if seg.comp_location_in_seconds >= time and\
                seg.comp_location_in_seconds < time + duration:
                return False
            # or, ends in range
            elif seg.comp_location_in_seconds + seg.duration_in_seconds >= time and\
                seg.comp_location_in_seconds + seg.duration_in_seconds < time + duration:
                return False
            # or, spans entire range
            elif seg.comp_location_in_seconds < time and\
                seg.comp_location_in_seconds + seg.duration_in_seconds >= time + duration:
                return False
        return True

    def contract(self, time, duration, min_contraction=0.0):
        """Remove empty gaps from the composition starting at a given
        time for a given duration.

        

        """

        # remove audio from the composition starting at time
        # for duration

        contract_dur = 0.0
        contract_start = time

        if self.empty_over_span(time, duration):
            contract_dur = duration
            contract_start = time
        else:
            starts = [s.comp_location_in_seconds for s in self.segments]
            ends = [s.comp_location_in_seconds + s.duration_in_seconds
                    for s in self.segments]

            key_starts = []
            key_ends = []

            for start in starts:
                if start >= time and start < time + duration:
                    # does a segment cover the location right before this start?
                    is_key_start = True
                    for seg in self.segments:
                        if seg.comp_location_in_seconds < start and\
                            seg.comp_location_in_seconds + seg.duration_in_seconds >= start:
                            is_key_start = False
                            break
                    if is_key_start:
                        key_starts.append(start)

            for end in ends:
                if end >= time and end < time + duration:
                    # does a segment cover the location right before this start?
                    is_key_end = True
                    for seg in self.segments:
                        if seg.comp_location_in_seconds <= end and\
                            seg.comp_location_in_seconds + seg.duration_in_seconds > end:
                            is_key_end = False
                            break
                    if is_key_end:
                        key_ends.append(end)

            if len(key_starts) + len(key_ends) == 0: return 0, 0

            # combine key starts and key ends
            key_both = [s for s in key_starts]
            key_both.extend([s for s in key_ends])
            key_both = sorted(key_both)

            first_key = key_both[0]
            if first_key in key_starts:
                contract_start = time
                contract_dur = first_key - time
            else:
                contract_start = first_key
                if len(key_both) >= 2:
                    contract_dur = key_both[1] - first_key
                else:
                    contract_dur = time + duration - first_key

        if contract_dur > min_contraction:
            for seg in self.segments:
                if seg.comp_location_in_seconds > contract_start:
                    dur_samples = int(seg.samplerate * contract_dur)
                    seg.comp_location -= dur_samples
            for dyn in self.dynamics:
                if dyn.comp_location_in_seconds > contract_start:
                    dur_samples = int(seg.samplerate * contract_dur)
                    dyn.comp_location -= dur_samples
            return contract_start, contract_dur
        else:
            return 0.0, 0.0

    def add_music_cue(self, track, comp_cue, song_cue, duration=6.0,
                      padding_before=12.0, padding_after=12.0):
        """Add a music cue to the composition. This doesn't do any audio
        analysis, it just aligns a specified point in the track
        (presumably music) with a location in the composition. See
        UnderScore_ for a visualization of what this is doing to the
        music track.

        .. _UnderScore: http://vis.berkeley.edu/papers/underscore/

        :param track: Track to align in the composition
        :type track: :py:class:`radiotool.composer.Track`
        :param float comp_cue: Location in composition to align music cue (in seconds)
        :param float song_cue: Location in the music track to align with the composition cue (in seconds)
        :param float duration: Duration of music after the song cue before the music starts to fade out (in seconds)
        :param float padding_before: Duration of music playing softly before the music cue/composition cue (in seconds)
        :param float padding_after: Duration of music playing softly after the music cue/composition cue (in seconds)
        """

        self.tracks.add(track)
        
        pre_fade = 3
        post_fade = 3
        
        if padding_before + pre_fade > song_cue:
            padding_before = song_cue - pre_fade
            
        if padding_before + pre_fade > score_cue:
            padding_before = score_cue - pre_fade

        s = Segment(track, score_cue - padding_before - pre_fade,
                    song_cue - padding_before - pre_fade,
                    pre_fade + padding_before + duration +
                    padding_after + post_fade)

        self.add_segment(s)
        
        d = []

        dyn_adj = 1
        
        track.current_frame = 0
        
        d.append(Fade(track, score_cue - padding_before - pre_fade, pre_fade,
                      0, .1*dyn_adj, fade_type="linear"))

        d.append(Fade(track, score_cue - padding_before, padding_before,
                      .1*dyn_adj, .4*dyn_adj, fade_type="exponential"))

        d.append(Volume(track, score_cue, duration, .4*dyn_adj))

        d.append(Fade(track, score_cue + duration, padding_after,
                      .4*dyn_adj, 0, fade_type="exponential"))

        d.append(Fade(track, score_cue + duration + padding_after, post_fade,
                      .1*dyn_adj, 0, fade_type="linear"))
        self.add_dynamics(d)
    
    def _remove_end_silence(self, frames):
        subwindow_n_frames = int(1/16.0 * min(s.samplerate for s in self.tracks))

        segments = segment_array(frames, subwindow_n_frames, overlap=.5)

        volumes = np.apply_along_axis(RMS_energy, 1, segments)

        min_subwindow_vol = min(np.sum(np.abs(segments), 1) /\
                            subwindow_n_frames)
        min_subwindow_vol = min(volumes)

        # some threshold? what if there are no zeros?
    
        min_subwindow_vol_index = np.where(volumes <= 2.0 *
                                          min_subwindow_vol)
    
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
    
    def duration(self):
        return max([x.comp_location + x.duration
                    for x in self.segments])

    def duration_in_seconds(self):
        return max([x.comp_location_in_seconds + x.duration_in_seconds
                    for x in self.segments])

    def build(self, track_list=None, adjust_dynamics=False,
        min_length=None, channels=None):
        """
        Create a numpy array from the composition.

        :param track_list: List of tracks to include in composition generation (``None`` means all tracks will be used)
        :type track_list: list of :py:class:`radiotool.composer.Track`
        :param int min_length: Minimum length of output array (in frames). Will zero pad extra length.
        :param bool. adjust_dynamics: Automatically adjust dynamics. Will document later.
        """
        if track_list is None:
            track_list = self.tracks

        if channels is None:
            channels = self.channels

        parts = {}
        starts = {}
        
        # for universal volume adjustment
        all_frames = np.array([])
        song_frames = np.array([])
        speech_frames = np.array([])
        
        longest_part = max([x.comp_location + x.duration
                            for x in self.segments])
        if len(self.dynamics) > 0:
            longest_part = max((longest_part,
                max([x.comp_location + x.duration
                     for x in self.dynamics])))
        
        for track_idx, track in enumerate(track_list):
            segments = sorted([v for v in self.segments if v.track == track], 
                              key=lambda k: k.comp_location + k.duration)
            dyns = sorted([d for d in self.dynamics if d.track == track],
                           key=lambda k: k.comp_location)

            if len(segments) > 0:
                start_loc = min([x.comp_location for x in segments])
                end_loc = max([x.comp_location + x.duration for x in segments])
                if len(dyns) > 0:
                    start_loc = min((start_loc, 
                         min([d.comp_location for d in dyns])))
                    end_loc = max((end_loc,
                        max([d.comp_location + d.duration for d in dyns])))
                
                starts[track] = start_loc

                parts[track] = np.zeros((end_loc - start_loc, channels))
                
                for s in segments:

                    frames = s.get_frames(channels=channels).\
                        reshape(-1, channels)
                    
                    # for universal volume adjustment
                    if adjust_dynamics:
                        all_frames = np.append(all_frames,
                            self._remove_end_silence(frames.flatten()))
                        if isinstance(track, Song):
                            song_frames = np.append(song_frames,
                                self._remove_end_silence(frames.flatten()))
                        elif isinstance(track, Speech):
                            speech_frames = np.append(speech_frames,
                                self._remove_end_silence(frames.flatten()))
                                
                    parts[track][s.comp_location - start_loc:
                                 s.comp_location - start_loc + s.duration,
                                 :] = frames

            for d in dyns:
                vol_frames = d.to_array(channels)

                parts[track][d.comp_location - start_loc :
                             d.comp_location - start_loc + d.duration,
                             :] *= vol_frames


        if adjust_dynamics:
            total_energy = RMS_energy(all_frames)
            song_energy = RMS_energy(song_frames)
            speech_energy = RMS_energy(speech_frames)
                
        # dyn_adj = 0.10 / total_energy
        # dyn_adj = speech_energy / sqrt(song_energy) * 5
        if adjust_dynamics:
            if not np.isnan(speech_energy) and not np.isnan(song_energy):
                dyn_adj = sqrt(speech_energy / song_energy) * 1.15
            else:
                dyn_adj = 1
        else:
            dyn_adj = 1

        if longest_part < min_length:
            longest_part = min_length
        out = np.zeros((longest_part, channels))
        for track, part in parts.iteritems():
            out[starts[track]:starts[track] + len(part)] += part

        return out
    
    def export(self, **kwargs):
        """
        Generate audio file from composition.

        :param str. filename: Output filename (no extension)
        :param str. filetype: Output file type (only .wav supported for now)
        :param integer samplerate: Sample rate of output audio
        :param integer channels: Channels in output audio, if different than originally specified
        :param bool. separate_tracks: Also generate audio file for each track in composition
        :param int min_length: Minimum length of output array (in frames). Will zero pad extra length.
        :param bool. adjust_dynamics: Automatically adjust dynamics (will document later)

        """
        # get optional args
        filename = kwargs.pop('filename', 'out')
        filetype = kwargs.pop('filetype', 'wav')
        adjust_dynamics = kwargs.pop('adjust_dynamics', False)
        samplerate = kwargs.pop('samplerate', None)
        channels = kwargs.pop('channels', self.channels)
        separate_tracks = kwargs.pop('separate_tracks', False)
        min_length = kwargs.pop('min_length', None)
        
        if samplerate is None:
            samplerate = np.min([track.samplerate for track in self.tracks])

        encoding = 'pcm16'
        to_mp3 = False
        if filetype == 'ogg':
            encoding = 'vorbis'
        elif filetype == 'mp3':
            filetype = 'wav'
            to_mp3 = True

        if separate_tracks:
            # build the separate parts of the composition if desired
            for track in self.tracks:
                out = self.build(track_list=[track],
                                 adjust_dynamics=adjust_dynamics,
                                 min_length=min_length,
                                 channels=channels)
                out_file = Sndfile("%s-%s.%s" %
                                   (filename, track.name, filetype),
                                   'w',
                                   Format(filetype, encoding=encoding),
                                   channels, samplerate)
                out_file.write_frames(out)
                out_file.close()

        # always build the complete composition
        out = self.build(adjust_dynamics=adjust_dynamics,
                         min_length=min_length,
                         channels=channels)

        out_filename = "%s.%s" % (filename, filetype)
        out_file = Sndfile(out_filename, 'w',
                           Format(filetype, encoding=encoding), 
                           channels, samplerate)
        out_file.write_frames(out)
        out_file.close()

        if LIBXMP and filetype == "wav":
            xmp = libxmp.XMPMeta()
            ns = libxmp.consts.XMP_NS_DM
            p = xmp.get_prefix_for_namespace(ns)
            xpath = p + 'Tracks'
            xmp.append_array_item(ns, xpath, None, 
                array_options={"prop_value_is_array": True},
                prop_value_is_struct=True)

            xpath += '[1]/' + p
            xmp.set_property(ns, xpath + "trackName", "CuePoint Markers")
            xmp.set_property(ns, xpath + "trackType", "Cue")
            xmp.set_property(ns, xpath + "frameRate", "f%d" % samplerate)

            for i, lab in enumerate(self.labels):
                xmp.append_array_item(ns, xpath + "markers", None,
                    array_options={"prop_value_is_array": True},
                    prop_value_is_struct=True)
                xmp.set_property(ns,
                    xpath + "markers[%d]/%sname" % (i + 1, p), lab.name)
                xmp.set_property(ns,
                    xpath + "markers[%d]/%sstartTime" % (i + 1, p),
                    str(lab.sample(samplerate))) 

            xmpfile = libxmp.XMPFiles(file_path=out_filename, open_forupdate=True)
            if xmpfile.can_put_xmp(xmp):
                xmpfile.put_xmp(xmp)
            xmpfile.close_file()

        if to_mp3:
            wav_to_mp3(out_filename, delete_wav=True)

        return out
