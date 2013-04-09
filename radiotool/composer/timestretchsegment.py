class TimeStretchSegment(Segment):
    #from scipy.signal import resample

    def __init__(self, track, score_location, start, orig_duration, new_duration):
        Segment.__init__(self, track, score_location, start, new_duration)
        self.orig_duration = int(orig_duration * self.samplerate)

    def get_frames(self, channels=2):
        self.track.set_frame(self.start)
        frames = self.track.read_frames(self.orig_duration)
        frames = resample(frames, self.duration)
        self.track.set_frame(0)
        return frames