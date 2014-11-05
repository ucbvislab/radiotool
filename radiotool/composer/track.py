import re
import os.path
import subprocess

from scikits.audiolab import Sndfile
import numpy as np
try:
    import libxmp
    import libxmp.utils
    LIBXMP = True
except:
    LIBXMP = False

from ..utils import zero_crossing_first, zero_crossing_last
from label import Label


class Track(object):
    """Represents a wrapped .wav file."""

    def __init__(self, fn, name="No name", labels=None, labels_in_file=False):
        """Create a Track object

        :param str. fn: Path to audio file (wav preferred, mp3 ok)
        :param str. name: Name of track
        """
        self.filename = fn
        self.name = name

        (base, extension) = os.path.splitext(self.filename)
        if extension == ".mp3":
            try:
                print "Creating wav from {}".format(self.filename)
                new_fn = base + '.wav'
                subprocess.check_output("lame --decode \"{}\" \"{}\"".format(
                    self.filename, new_fn), shell=True)
                self.filename = new_fn
            except:
                print "Could not create wav from mp3"
                raise

        self.sound = Sndfile(self.filename, 'r')
        self.current_frame = 0
        self.channels = self.sound.channels

        if labels is not None and labels_in_file:
            raise Exception(
                "Must only define one of labels and labels_in_file")
        if labels_in_file and not LIBXMP:
            raise Exception(
                "Cannot use labels_in_file without python-xmp-toolkit")
        if labels_in_file and LIBXMP:
            self.labels = self._extract_labels(fn)
        else:
            self.labels = labels

    def read_frames(self, n, channels=None):
        """Read ``n`` frames from the track, starting
        with the current frame

        :param integer n: Number of frames to read
        :param integer channels: Number of channels to return (default
            is number of channels in track)
        :returns: Next ``n`` frames from the track, starting with ``current_frame``
        :rtype: numpy array
        """
        if channels is None:
            channels = self.channels

        if channels == 1:
            out = np.zeros(n)
        elif channels == 2:
            out = np.zeros((n, 2))
        else:
            print "Input needs to be 1 or 2 channels"
            return
        if n > self.remaining_frames():
            print "Trying to retrieve too many frames!"
            print "Asked for", n
            n = self.remaining_frames()
            print "Returning", n

        if self.channels == 1 and channels == 1:
            out = self.sound.read_frames(n)
        elif self.channels == 1 and channels == 2:
            frames = self.sound.read_frames(n)
            out = np.vstack((frames.copy(), frames.copy())).T
        elif self.channels == 2 and channels == 1:
            frames = self.sound.read_frames(n)
            out = np.mean(frames, axis=1)
        elif self.channels == 2 and channels == 2:
            out[:n, :] = self.sound.read_frames(n)

        self.current_frame += n

        return out

    @property
    def current_frame(self):
        """Get and set the current frame of the track"""
        return self._current_frame

    @current_frame.setter
    def current_frame(self, n):
        """Sets current frame to ``n``

        :param integer n: Frame to set to ``current_frame``
        """
        self.sound.seek(n)
        self._current_frame = n

    def reset(self):
        """Sets current frame to 0
        """
        self.current_frame = 0

    def all_as_mono(self):
        """Get the entire track as 1 combined channel

        :returns: Track frames as 1 combined track
        :rtype: 1d numpy array
        """
        return self.range_as_mono(0, self.duration)

    def range_as_mono(self, start_sample, end_sample):
        """Get a range of frames as 1 combined channel

        :param integer start_sample: First frame in range
        :param integer end_sample: Last frame in range (exclusive)
        :returns: Track frames in range as 1 combined channel
        :rtype: 1d numpy array of length ``end_sample - start_sample``
        """
        tmp_current = self.current_frame
        self.current_frame = start_sample
        tmp_frames = self.read_frames(end_sample - start_sample)
        if self.channels == 2:
            frames = np.mean(tmp_frames, axis=1)
        elif self.channels == 1:
            frames = tmp_frames
        else:
            raise IOError("Input audio must have either 1 or 2 channels")
        self.current_frame = tmp_current
        return frames

    @property
    def samplerate(self):
        """Get the sample rate of the track"""
        return self.sound.samplerate

    def remaining_frames(self):
        """Get the number of frames remaining in the track"""
        return self.sound.nframes - self.current_frame

    @property
    def duration(self):
        """Get the duration of total frames in the track"""
        return self.sound.nframes

    @property
    def duration_in_seconds(self):
        """Get the duration of the track in seconds"""
        "Should not set track length"
        return self.duration / float(self.samplerate)

    def loudest_time(self, start=0, duration=0):
        """Find the loudest time in the window given by start and duration
        Returns frame number in context of entire track, not just the window.

        :param integer start: Start frame
        :param integer duration: Number of frames to consider from start
        :returns: Frame number of loudest frame
        :rtype: integer
        """
        if duration == 0:
            duration = self.sound.nframes
        self.current_frame = start
        arr = self.read_frames(duration)
        # get the frame of the maximum amplitude
        # different names for the same thing...
        # max_amp_sample = a.argmax(axis=0)[a.max(axis=0).argmax()]
        max_amp_sample = int(np.floor(arr.argmax()/2)) + start
        return max_amp_sample

    def refine_cut(self, cut_point, window_size=1):
        return cut_point

    def zero_crossing_before(self, n):
        """Find nearest zero crossing in waveform before frame ``n``"""
        n_in_samples = int(n * self.samplerate)

        search_start = n_in_samples - self.samplerate
        if search_start < 0:
            search_start = 0

        frame = zero_crossing_last(
            self.range_as_mono(search_start, n_in_samples)) + search_start

        return frame / float(self.samplerate)

    def zero_crossing_after(self, n):
        """Find nearest zero crossing in waveform after frame ``n``"""
        n_in_samples = int(n * self.samplerate)
        search_end = n_in_samples + self.samplerate
        if search_end > self.duration:
            search_end = self.duration

        frame = zero_crossing_first(
            self.range_as_mono(n_in_samples, search_end)) + n_in_samples

        return frame / float(self.samplerate)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            self._labels = None
        else:
            self._labels = sorted(labels, key=lambda x: x.time)

    def label(self, t):
        """Get the label of the song at a given time in seconds
        """
        if self.labels is None:
            return None
        prev_label = None
        for l in self.labels:
            if l.time > t:
                break
            prev_label = l
        if prev_label is None:
            return None
        return prev_label.name

    def _extract_labels(self, filename):
        if not LIBXMP:
            return None

        xmp = libxmp.utils.file_to_dict(filename)
        meta = libxmp.XMPMeta()
        ns = libxmp.consts.XMP_NS_DM
        p = meta.get_prefix_for_namespace(ns)

        #track_re = re.compile("^" + p + r"Tracks\[(\d+)\]$")
        #n_tracks = 0
        cp_track = None
        new_xmp = {}
        for prop in xmp[ns]:
            new_xmp[prop[0]] = prop[1:]

        # find the cuepoint markers track
        name_re = re.compile("^" + p + r"Tracks\[(\d+)\]/" + p + "trackName$")
        for prop, val in new_xmp.iteritems():
            match = name_re.match(prop)
            if match:
                if val[0] == "CuePoint Markers":
                    cp_track = match.group(1)

        # get all the markers from it
        cp_path = re.compile(r"^%sTracks\[%s\]/%smarkers\[(\d+)\]$" %
                             (p, cp_track, p))
        markers = []
        sr = float(new_xmp["%sTracks[%s]/%sframeRate" %
                   (p, cp_track, p)][0].replace('f', ''))

        for prop, val in new_xmp.iteritems():
            match = cp_path.match(prop)
            if match:
                markers.append(Label(
                    new_xmp[prop + '/' + p + 'name'][0],
                    float(new_xmp[prop + '/' + p + 'startTime'][0]) / sr))

        if len(markers) is 0:
            return None
        return markers
