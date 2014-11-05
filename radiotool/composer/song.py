import hashlib
import pickle
import os

from ..algorithms import librosa_analysis
from track import Track

class Song(Track):
    """A :py:class:`radiotool.composer.Track`
    subclass that wraps a music .wav file.
    Allows access to a musical analysis of the song.
    """

    def __init__(self, fn, name="Song name", cache_dir=None,
                 refresh_cache=False, labels=None, labels_in_file=False):
        self._analysis = None
        self._checksum = None
        self.refresh_cache = refresh_cache
        self.cache_dir = cache_dir

        Track.__init__(self, fn, name, labels=labels,
                       labels_in_file=labels_in_file)

    @property
    def analysis(self):
        """Get musical analysis of the song using the librosa library
        """
        if self._analysis is not None:
            return self._analysis

        if self.cache_dir is not None:
            path = os.path.join(self.cache_dir, self.checksum)
            try:
                if self.refresh_cache: raise IOError
                with open(path + '.pickle', 'rb') as pickle_file:
                    self._analysis = pickle.load(pickle_file)
            except IOError:
                self._analysis = librosa_analysis.analyze_frames(self.all_as_mono(), self.samplerate)
                with open(path + '.pickle', 'wb') as pickle_file:
                    pickle.dump(self._analysis, pickle_file, pickle.HIGHEST_PROTOCOL)
        else:
            self._analysis = librosa_analysis.analyze_frames(self.all_as_mono(), self.samplerate)
        return self._analysis

    def features_cached(self):
        if self.cache_dir is not None:
            path = os.path.join(self.cache_dir, self.checksum)
            try:
                if self.refresh_cache: raise IOError
                with open(path + '.pickle', 'rb') as pickle_file:
                    return True
            except IOError:
                pass
        return False

    @property
    def checksum(self):
        if self._checksum is not None:
            return self._checksum

        block_size = 65536
        hasher = hashlib.sha256()
        with open(self.filename, 'rb') as f:
            buf = f.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(block_size)
        self._checksum = hasher.hexdigest()
        return self._checksum

