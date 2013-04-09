class RawVolume(Dynamic):
    def __init__(self, segment, volume_frames):
        self.track = segment.track
        self.samplerate = segment.track.samplerate()
        self.score_location = segment.score_location
        self.duration = segment.duration
        self.volume_frames = volume_frames
        if self.duration != len(volume_frames):
            raise Exception("Duration must be same as volume frame length")
    
    def to_array(self, channels=2):
        if channels == 1:
            return self.volume_frames.reshape(-1, 1)
        if channels == 2:
            return N.tile(self.volume_frames, (1, 2))
        raise Exception(
            "RawVolume doesn't know what to do with %s channels" % channels)
        
