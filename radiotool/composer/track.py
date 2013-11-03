from scikits.audiolab import Sndfile, Format
import numpy as N

from ..utils import zero_crossing_first, zero_crossing_last

class Track(object):
    """Represents a wrapped .wav file."""

    def __init__(self, fn, name="No name"):
        """Create a Track object

        :param str. fn: Path to wav file
        :param str. name: Name of track
        """
        self.filename = fn
        self.name = name

        self.sound = Sndfile(self.filename, 'r')
        self.current_frame = 0
        self.channels = self.sound.channels


    def read_frames(self, n):
        """Read ``n`` frames from the track, starting
        with the current frame

        :param integer n: Number of frames to read
        :returns: Next ``n`` frames from the track, starting with ``current_frame``
        :rtype: numpy array
        """

        if self.channels == 1:
            out = N.zeros(n)
        elif self.channels == 2:
            out = N.zeros((n, 2))
        else:
            print "Input needs to have 1 or 2 channels"
            return
        if n > self.remaining_frames():
            print "Trying to retrieve too many frames!"
            print "Asked for", n
            n = self.remaining_frames()

        if self.channels == 1:
            out = self.sound.read_frames(n)
        elif self.channels == 2:
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
            frames = N.mean(tmp_frames, axis=1)
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
        max_amp_sample = int(N.floor(arr.argmax()/2)) + start
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
