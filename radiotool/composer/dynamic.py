class Dynamic:
    def __init__(self, track, score_location, duration):
        self.track = track
        self.samplerate = track.samplerate()
        self.score_location = int(round(score_location * self.samplerate))
        self.duration = int(round(duration * self.samplerate))
        
    def to_array(self, channels=2):
        return N.ones( (self.duration, channels) )
        
    def __str__(self):
        return "Dynamic at %d with duration %d" % (self.score_location,
                                                   self.duration)
        