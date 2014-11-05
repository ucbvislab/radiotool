import numpy as np

class Dynamic(object):
    """(Abstract) volume control for tracks"""
    def __init__(self, track, comp_location, duration):
        self.track = track
        self.samplerate = track.samplerate
        self.comp_location_in_seconds = comp_location
        self.duration_in_seconds = duration
        
    def to_array(self, channels=2):
        return np.ones( (self.duration, channels) )
        
    def __str__(self):
        return "Dynamic at %d with duration %d" % (self.comp_location,
                                                   self.duration)
    
    @property
    def duration_in_seconds(self):
        return self.duration / float(self.samplerate)

    @duration_in_seconds.setter
    def duration_in_seconds(self, duration_in_seconds):
        self.duration = int(duration_in_seconds * self.samplerate)

    @property
    def comp_location_in_seconds(self):
        return self.comp_location / float(self.samplerate)

    @comp_location_in_seconds.setter
    def comp_location_in_seconds(self, comp_location_in_seconds):
        self.comp_location = int(comp_location_in_seconds * self.samplerate)
