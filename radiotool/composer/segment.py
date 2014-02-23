import numpy as N

class Segment(object):
    """A slice of a :py:class:`radiotool.composer.Track`
    """

    def __init__(self, track, comp_location, start, duration):
        """Create a segment from a track.

        The segment represents part of a track (from ``start``, for
        ``duration`` seconds) and will be played at location
        ``comp_location`` when added to a composition.

        :param track: Track to create segment from
        :type track: :py:class:`radiotool.composer.Track`
        :param float comp_location: Location in composition to play this segment (in seconds)
        :param float start: Start of segment (in seconds)
        :param float duration: Duration of segment (in seconds)

        """
        self.samplerate = track.samplerate
        self.track = track
        self.comp_location_in_seconds = comp_location
        self.start_in_seconds = start
        self.duration_in_seconds = duration

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration
        self._duration_in_seconds = duration / float(self.samplerate)

    @property
    def duration_in_seconds(self):
        return self._duration_in_seconds

    @duration_in_seconds.setter
    def duration_in_seconds(self, duration_in_seconds):
        self._duration_in_seconds = duration_in_seconds
        self._duration = int(duration_in_seconds * self.samplerate)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start
        self._start_in_seconds = start / float(self.samplerate)

    @property
    def start_in_seconds(self):
        return self._start_in_seconds

    @start_in_seconds.setter
    def start_in_seconds(self, start_in_seconds):
        self._start_in_seconds = start_in_seconds
        self._start = int(start_in_seconds * self.samplerate)

    @property
    def comp_location(self):
        return self._comp_location

    @comp_location.setter
    def comp_location(self, comp_location):
        self._comp_location = comp_location
        self._comp_location_in_seconds = comp_location / float(self.samplerate)

    @property
    def comp_location_in_seconds(self):
        return self._comp_location_in_seconds

    @comp_location_in_seconds.setter
    def comp_location_in_seconds(self, comp_location_in_seconds):
        self._comp_location_in_seconds = comp_location_in_seconds
        self._comp_location = int(comp_location_in_seconds * self.samplerate)

    def get_frames(self, channels=2):
        """Get numpy array of frames corresponding to the segment.

        :param integer channels: Number of channels in output array
        :returns: Array of frames in the segment
        :rtype: numpy array

        """
        tmp_frame = self.track.current_frame
        self.track.current_frame = self.start
        frames = self.track.read_frames(self.duration, channels=channels)
        self.track.current_frame = tmp_frame

        return frames.copy()
