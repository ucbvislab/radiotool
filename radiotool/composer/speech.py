import numpy as np

from track import Track
from ..utils import segment_array, RMS_energy

class Speech(Track):
    """A :py:class:`radiotool.composer.Track` 
    subclass that wraps a speech .wav file"""

    def __init__(self, fn, name="Speech name", labels=None, labels_in_file=False):
        Track.__init__(self, fn, name, labels=labels,
                       labels_in_file=labels_in_file)
    
    def refine_cut(self, cut_point, window_size=1):
        cut_point = max(.5 * window_size, cut_point)

        cf = self.current_frame
        self.current_frame = max(int((cut_point - window_size / 2.0) * self.samplerate), 0)
        frames = self.read_frames(window_size * self.samplerate, channels=1)
        subwindow_n_frames = int((window_size / 16.0) * self.samplerate)
        self.current_frame = cf

        segments = segment_array(frames, subwindow_n_frames, overlap=.5)

        # segments = segments.reshape((-1, subwindow_n_frames * 2))

        volumes = np.apply_along_axis(RMS_energy, 1, segments)
 
        min_subwindow_vol = min(np.sum(np.abs(segments), 1) / subwindow_n_frames)
        min_subwindow_vol = min(volumes)

        min_subwindow_vol_index = np.where(volumes <= 1.1 *
                                          min_subwindow_vol)

        
        # find longest span of "silence" and set to the beginning
        # adapted from 
        # http://stackoverflow.com/questions/3109052/
        # find-longest-span-of-consecutive-array-keys
        last_key = -1
        cur_list = []
        long_list = []
        for idx in min_subwindow_vol_index[0]:
            if idx != last_key + 1:
                cur_list = []
            cur_list.append(idx)
            if(len(cur_list) > len(long_list)):
                long_list = cur_list
            last_key = idx
        
        new_cut_point = (self.samplerate * (cut_point - window_size / 2.0) + 
                         (long_list[0] + 1) * 
                         int(subwindow_n_frames / 2.0))
        # print "first min subwindow", long_list[0], "total", len(volumes)

        print "{} -> {}".format(cut_point, round(new_cut_point / self.samplerate, 2))

        return round(new_cut_point / self.samplerate, 2)
        # have to add the .5 elsewhere to get that effect!
        