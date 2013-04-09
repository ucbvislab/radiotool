from dynamic import Dynamic

class Volume(Dynamic):
    def __init__(self, track, score_location, duration, volume):
        Dynamic.__init__(self, track, score_location, duration)
        self.volume = volume
        
    def to_array(self, channels=2):
        return N.linspace(self.volume, self.volume, 
            self.duration * channels).reshape(self.duration, channels)