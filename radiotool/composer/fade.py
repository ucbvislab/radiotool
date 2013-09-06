import numpy as N

from dynamic import Dynamic

class Fade(Dynamic):
    """Create a fade dynamic in a composition"""

    def __init__(self, track, comp_location, duration, 
                in_volume, out_volume, fade_type="linear"):
        """A fade is a :py:class:`radiotool.composer.Dynamic` that
        represents a fade in a track (either in or out).
        
        Currently supported fade types are ``linear`` and
        ``exponential``.

        The exponential fades are probably a bit quirky, but they work
        for me for some use cases.

        :param track: Track to fade
        :type track: :py:class:`radiotool.composer.Track`
        :param float comp_location: Location in composition to start fade (in seconds)
        :param float duration: Duration of fade (in seconds)
        :param float in_volume: Initial volume multiplier
        :param float out_volume: Ending volume multiplier
        :param string fade_type: Type of fade (``linear`` or ``exponential``)

        """        
        Dynamic.__init__(self, track, comp_location, duration)
        self.in_volume = in_volume
        self.out_volume = out_volume
        self.fade_type = fade_type
        
    def to_array(self, channels=2):
        """Generate the array of volume multipliers for the dynamic"""
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