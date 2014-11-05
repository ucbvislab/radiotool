class Segment(object):
    """A slice of a :py:class:`radiotool.composer.Track`
    """

    def __init__(self, track, comp_location, start, duration, effects=None):
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
        if effects is None:
            self.effects = []
        else:
            self.effects = effects

    @property
    def duration_in_seconds(self):
        return self.duration / float(self.samplerate)

    @duration_in_seconds.setter
    def duration_in_seconds(self, duration_in_seconds):
        self.duration = int(duration_in_seconds * self.samplerate)

    @property
    def start_in_seconds(self):
        return self.start / float(self.samplerate)

    @start_in_seconds.setter
    def start_in_seconds(self, start_in_seconds):
        self.start = int(start_in_seconds * self.samplerate)

    @property
    def comp_location_in_seconds(self):
        return self.comp_location / float(self.samplerate)

    @comp_location_in_seconds.setter
    def comp_location_in_seconds(self, comp_location_in_seconds):
        self.comp_location = int(comp_location_in_seconds * self.samplerate)

    def add_effect(self, effect):
        self.effects.append(effect)

    def add_effects(self, effects):
        self.effects.extend(effects)

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

        for effect in self.effects:
            frames = effect.apply_to(frames, self.samplerate)

        return frames.copy()
