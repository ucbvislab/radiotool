import numpy as N

from dynamic import Dynamic

class Volume(Dynamic):
    """Adjust the volume of a track in a composition"""

    def __init__(self, track, comp_location, duration, volume):
        """Create a dynamic to adjust the volume of a track in a
        composition.

        Any segments in the composition that include the ``track``
        between ``comp_location`` and ``comp_location + duration``
        will be adjust to the given ``volume``. Here, ``volume`` is a
        constant multiplier (0.0 for zero volume, 1.0 for normal
        volume). You can use a range of volumes, but obviously if
        volume is much greater than 1.0, there will likely be clipping
        in the final composition.

        :param track: Track whose volume to adjust
        :type track: :py:class:`radiotool.composer.Track`
        :param float comp_location: Location to begin volume adjustment in composition (in seconds)
        :param float duration: Duration of volume adjustment (in seconds)
        :param float volume: Volume throughout the duration of the adjustment (1.0: normal, to 0.0: muted)
        """        
        Dynamic.__init__(self, track, comp_location, duration)
        self.volume = volume
        
    def to_array(self, channels=2):
        """Generate the array of multipliers for the dynamic"""
        return N.linspace(self.volume, self.volume, 
            self.duration * channels).reshape(self.duration, channels)