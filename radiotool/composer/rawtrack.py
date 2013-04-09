import numpy as N

from track import Track


class RawTrack(Track):

    def __init__(self, frames, name="Raw frames name", samplerate=44100):
        self._sr = samplerate
        self.frames = frames
        self.name = name
        self.filename = "RAW_" + name
        try:
            self.channels = N.shape(frames)[1]
        except:
            self.channels = 1
        self.current_frame = 0
        self._total_frames = N.shape(frames)[0]
    
    def samplerate(self):
        return self._sr
    
    def sr(self):
        return self._sr
    
    def set_frame(self, n):
        self.current_frame = n
    
    def total_frames(self):
        return self._total_frames
    
    def remaining_frames(self):
        return self._total_frames - self.current_frame
    
    def reset(self):
        self.current_frame = 0
    
    def read_frames(self, n):
        if self.channels == 1:
            out = N.zeros(n)
        elif self.channels == 2:
            out = N.zeros((n, 2))
        else:
            print "Input needs to have 1 or 2 channels"
            return
        if n > self.remaining_frames():
            print "Trying to retrieve too many frames!"
            print "Asked for", n
            n = self.remaining_frames()

        if self.channels == 1:
            out = self.frames[self.current_frame:self.current_frame + n]
        elif self.channels == 2:
            out[:n, :] = self.frames[
                self.current_frame:self.current_frame + n, :]

        self.current_frame += n
        return out