import numpy as N

from dynamic import Dynamic

class RawVolume(Dynamic):
    """Dynamic with manually-specified volume multiplier array"""

    def __init__(self, segment, volume_frames):
        """Create a dynamic that manually specifies the volume
        multiplier array.
        
        :param segment: Segment for which to create dynamic
        :type segment: :py:class:`radiotool.composer.Segment`
        :param volume_frames: Raw volume multiplier frames
        :type volume_frames: numpy array
        """        
        self.track = segment.track
        self.samplerate = segment.track.samplerate
        self.comp_location = segment.comp_location
        self.duration = segment.duration
        self.volume_frames = volume_frames
        if self.duration != len(volume_frames):
            raise Exception("Duration must be same as volume frame length")
    
    def to_array(self, channels=2):
        """Return the array of multipliers for the dynamic"""
        if channels == 1:
            return self.volume_frames.reshape(-1, 1)
        if channels == 2:
            return N.tile(self.volume_frames, (1, 2))
        raise Exception(
            "RawVolume doesn't know what to do with %s channels" % channels)
        
