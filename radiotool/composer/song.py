
from ..algorithms import librosa_analysis
from track import Track

class Song(Track):
    """A :py:class:`radiotool.composer.Track`
    subclass that wraps a music .wav file.
    Allows access to a musical analysis of the song.
    """

    def __init__(self, fn, name="Song name"):
        self._analysis = None
        Track.__init__(self, fn, name)

    @property
    def analysis(self):
        """Get musical anaylsis of the song using the librosa library
        """
        if self._analysis is not None:
            return self._analysis

        self._analysis = librosa_analysis.analyze_frames(self.all_as_mono(), self.samplerate)
        return self._analysis

