import numpy as N

from track import Track


class RawTrack(Track):
    """A :py:class:`radiotool.composer.Track` subclass that wraps raw PCM
    data (as a numpy array).
    """

    def __init__(self, frames, name="Raw frames name", samplerate=44100):
        """Create a track with raw PCM data in array ``frames``

        :param frames: Raw PCM data array
        :type frames: numpy array
        :param integer samplerate: Sample rate of frames
        :param string name: Name of track

        """
        self._samplerate = samplerate
        self.frames = frames
        self.name = name
        self.filename = "RAW_" + name
        try:
            self.channels = N.shape(frames)[1]
        except:
            self.channels = 1
        self.current_frame = 0
        self._total_frames = N.shape(frames)[0]
    
    @property
    def samplerate(self):
        return self._samplerate

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, n):
        self._current_frame = n

    @property
    def duration(self):
        return self._total_frames

    def remaining_frames(self):
        return self._total_frames - self.current_frame
    
    def reset(self):
        self.current_frame = 0
    
    def read_frames(self, n, channels=None):
        if channels is None:
            channels = self.channels

        if channels == 1:
            out = N.zeros(n)
        elif channels == 2:
            out = N.zeros((n, 2))
        else:
            print "Input needs to have 1 or 2 channels"
            return
        if n > self.remaining_frames():
            print "Trying to retrieve too many frames!"
            print "Asked for", n
            n = self.remaining_frames()

        if self.channels == 1 and channels == 1:
            out = self.frames[self.current_frame:self.current_frame + n]
        elif self.channels == 1 and channels == 2:
            frames = self.frames[self.current_frame:self.current_frame + n]
            out[:n, :] = [frames.copy(), frames.copy()]
        elif self.channels == 2 and channels == 1:
            frames = self.frames[
                self.current_frame:self.current_frame + n, :]
            out = N.mean(frames, axis=1)
        elif self.channels == 2 and channels == 2:
            out[:n, :] = self.frames[
                self.current_frame:self.current_frame + n, :]

        self.current_frame += n
        return out