from track import Track
from ..utils import segment_array

class Speech(Track):
    """A :py:class:`radiotool.composer.Track` 
    subclass that wraps a speech .wav file"""

    def __init__(self, fn, name="Speech name"):
        Track.__init__(self, fn, name)
    
    def refine_cut(self, cut_point, window_size=1):
        self.current_frame = int((cut_point - window_size / 2.0) * self.sr())
        frames = self.read_frames(window_size * self.sr())
        subwindow_n_frames = int((window_size / 16.0) * self.sr())

        segments = segment_array(frames, subwindow_n_frames, overlap=.5)

        segments = segments.reshape((-1, subwindow_n_frames * 2))

        volumes = N.apply_along_axis(RMS_energy, 1, segments)
 
        min_subwindow_vol = min(N.sum(N.abs(segments), 1) / subwindow_n_frames)
        min_subwindow_vol = min(volumes)

        min_subwindow_vol_index = N.where(volumes <= 1.1 * 
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
        
        new_cut_point = (self.sr() * (cut_point - window_size / 2.0) + 
                         (long_list[0] + 1) * 
                         int(subwindow_n_frames / 2.0))
        print "first min subwindow", long_list[0], "total", len(volumes)
        return round(new_cut_point / self.sr(), 2)
        # have to add the .5 elsewhere to get that effect!
        