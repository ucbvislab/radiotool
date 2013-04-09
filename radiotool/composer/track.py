from scikits.audiolab import Sndfile, Format
import numpy as N

class Track:
    
    def __init__(self, fn, name="No name"):
        """Create a Track object from a music filename"""
        self.filename = fn
        self.name = name

        self.sound = Sndfile(self.filename, 'r')
        self.current_frame = 0
        self.channels = self.sound.channels


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
        self.current_frame += n
        
        if self.channels == 1:
            out = self.sound.read_frames(n)
        elif self.channels == 2:
            out[:n, :] = self.sound.read_frames(n)
        return out

    def set_frame(self, n):
        self.sound.seek(n)
        self.current_frame = n

    def reset(self):
        self.set_frame(0)
        self.current_frame = 0

    def all_as_mono(self):
        """Get the entire track as 1 combined channel"""
        return self.range_as_mono(0, self.total_frames())

    def range_as_mono(self, start_sample, end_sample):
        """Get a range of frames as 1 combined channel"""
        tmp_current = self.current_frame
        self.set_frame(start_sample)
        tmp_frames = self.read_frames(end_sample - start_sample)
        if self.channels == 2:
            frames = N.mean(tmp_frames, axis=1)
        elif self.channels == 1:
            frames = tmp_frames
        else:
            raise IOError("Input audio must have either 1 or 2 channels")
        self.set_frame(tmp_current)
        return frames

    def samplerate(self):
        return self.sound.samplerate
        
    def sr(self):
        return self.samplerate()

    def remaining_frames(self):
        return self.sound.nframes - self.current_frame
        
    def total_frames(self):
        return self.sound.nframes
    
    def duration(self):
        return self.total_frames() / float(self.samplerate())
        
    def loudest_time(self, start=0, duration=0):
        """Find the loudest time in the window given by start and duration
        Returns frame number in context of entire track, not just the window
        """
        if duration == 0:
            duration = self.sound.nframes
        self.set_frame(start)
        arr = self.read_frames(duration)
        # get the frame of the maximum amplitude
        # different names for the same thing...
        # max_amp_sample = a.argmax(axis=0)[a.max(axis=0).argmax()]
        max_amp_sample = int(N.floor(arr.argmax()/2)) + start
        return max_amp_sample
    
    def refine_cut(self, cut_point, window_size=1):
        return cut_point
        
    def zero_crossing_before(self, n):
        """n is in seconds, finds the first zero crossing before n seconds"""
        n_in_samples = int(n * self.samplerate())

        search_start = n_in_samples - self.samplerate() 
        if search_start < 0:
            search_start = 0

        frame = zero_crossing_last(
            self.range_as_mono(search_start, n_in_samples)) + search_start

        # frame = zero_crossing_before(self.all_as_mono(), n_in_samples)
        return frame / float(self.samplerate())

    def zero_crossing_after(self, n):
        n_in_samples = int(n * self.samplerate())
        search_end = n_in_samples + self.samplerate()
        if search_end > self.total_frames():
            search_end = self.total_frames()

        frame = zero_crossing_first(
            self.range_as_mono(n_in_samples, search_end)) + n_in_samples

        # frame = zero_crossing_after(self.all_as_mono(), n_in_samples)
        return frame / float(self.samplerate())
