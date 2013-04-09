class Fade(Dynamic):
    # linear, exponential, (TODO: cosine)
    def __init__(self, track, score_location, duration, 
                in_volume, out_volume, fade_type="linear"):
        Dynamic.__init__(self, track, score_location, duration)
        self.in_volume = in_volume
        self.out_volume = out_volume
        self.fade_type = fade_type
        
    def to_array(self, channels=2):
        if self.fade_type == "linear":
            return N.linspace(self.in_volume, self.out_volume, 
                self.duration * channels)\
                .reshape(self.duration, channels)
        elif self.fade_type == "exponential":
            if self.in_volume < self.out_volume:
                return (N.logspace(8, 1, self.duration * channels,
                    base=.5) * (
                        self.out_volume - self.in_volume) / 0.5 + 
                        self.in_volume).reshape(self.duration, channels)
            else:
                return (N.logspace(1, 8, self.duration * channels, base=.5
                    ) * (self.in_volume - self.out_volume) / 0.5 + 
                    self.out_volume).reshape(self.duration, channels)
        elif self.fade_type == "cosine":
            return