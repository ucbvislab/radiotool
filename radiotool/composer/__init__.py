"""
.. module:: composer
    :synopsis: Create audio by piecing together other audio files

.. moduleauthor:: Steve Rubin <srubin@cs.berkeley.edu>

"""

from .composition import Composition
from .track import Track
from .rawtrack import RawTrack
from .speech import Speech
from .segment import Segment
from .timestretchsegment import TimeStretchSegment
from .dynamic import Dynamic
from .fade import Fade
from .volume import Volume
from .rawvolume import RawVolume
from .label import Label
from .song import Song
from .volumebreakpoint import VolumeBreakpoint, VolumeBreakpoints
from .effect import NotchFilter
