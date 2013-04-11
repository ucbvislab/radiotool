import numpy as N

class Segment:
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
        self.comp_location = int(comp_location * self.samplerate)
        self.start = int(start * self.samplerate)
        self.duration = int(duration * self.samplerate)

    def get_frames(self, channels=2):
        """Get numpy array of frames corresponding to the segment.

        :param integer channels: Number of channels in output array
        :returns: Array of frames in the segment
        :rtype: numpy array

        """
        tmp_frame = self.track.current_frame
        self.track.current_frame = self.start
        frames = self.track.read_frames(self.duration)
        self.track.current_frame = tmp_frame
        
        if channels == self.track.channels:
            return frames.copy()
        elif channels == 2 and self.track.channels == 1:
            return N.hstack((frames.copy(), frames.copy()))
        elif channels == 1 and self.track.channels == 2:
            return N.mean(frames, axis=1)