import numpy as N

class Dynamic(object):
    """(Abstract) volume control for tracks"""
    def __init__(self, track, comp_location, duration):
        self.track = track
        self.samplerate = track.samplerate
        self.comp_location_in_seconds = comp_location
        self.duration_in_seconds = duration
        
    def to_array(self, channels=2):
        return N.ones( (self.duration, channels) )
        
    def __str__(self):
        return "Dynamic at %d with duration %d" % (self.comp_location,
                                                   self.duration)
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
        