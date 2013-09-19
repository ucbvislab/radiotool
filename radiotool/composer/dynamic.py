import numpy as N

class Dynamic:
    """(Abstract) volume control for tracks"""
    def __init__(self, track, comp_location, duration):
        self.track = track
        self.samplerate = track.samplerate
        self.comp_location = int(round(comp_location * self.samplerate))
        self.duration = int(round(duration * self.samplerate))
        
    def to_array(self, channels=2):
        return N.ones( (self.duration, channels) )
        
    def __str__(self):
        return "Dynamic at %d with duration %d" % (self.comp_location,
                                                   self.duration)
        