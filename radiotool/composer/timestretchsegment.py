from segment import Segment
from scipy.signal import resample

class TimeStretchSegment(Segment):
    """Like a :py:class:`radiotool.composer.Segment`, but stretches
    time to fit a specified duration.
    """

    def __init__(self, track, comp_location, start, orig_duration, new_duration):
        """Create a time-stetched segment. 

        It acts like a :py:class:`radiotool.composer.Segment` but you
        can specify the target duration. The segment will then
        resample its frames to meet this duration.

        :param track: Track to slice
        :type track: :py:class:`radiotool.composer.Track`
        :param float comp_location: Location in composition to play this segment (in seconds)
        :param float start: Start of segment (in seconds)
        :param float orig_duration: Original duration of segment (in seconds)
        :param float new_duration: Target (stretched) duration of segment (in seconds)
        """
        Segment.__init__(self, track, comp_location, start, new_duration)
        self.orig_duration = int(orig_duration * self.samplerate)

    def get_frames(self, channels=2):
        self.track.current_frame = self.start
        frames = self.track.read_frames(self.orig_duration, channels=channels)
        frames = resample(frames, self.duration)
        self.track.current_frame = 0
        return frames