import numpy as N

class Segment:
    # score_location, start, and duration all in seconds
    # -- may have to change later if this isn't accurate enough
    def __init__(self, track, score_location, start, duration):
        self.samplerate = track.samplerate()
        self.track = track
        self.score_location = int(score_location * self.samplerate)
        self.start = int(start * self.samplerate)
        self.duration = int(duration * self.samplerate)

    def get_frames(self, channels=2):
        self.track.set_frame(self.start)
        frames = self.track.read_frames(self.duration)
        self.track.set_frame(0)
        
        if channels == self.track.channels:
            return frames.copy()
        elif channels == 2 and self.track.channels == 1:
            return N.hstack((frames.copy(), frames.copy()))
        elif channels == 1 and self.track.channels == 2:
            return N.mean(frames, axis=1)