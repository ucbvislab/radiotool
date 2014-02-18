import hashlib
import pickle
import os
import re

try:
    import libxmp
    import libxmp.utils
    LIBXMP = True
except:
    LIBXMP = False

from ..algorithms import librosa_analysis
from track import Track
from label import Label

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

        if labels is not None and labels_in_file:
            raise Exception("Must only define one of labels and labels_in_file")
        if labels_in_file and not LIBXMP:
            raise Exception("Cannot use labels_in_file without python-xmp-toolkit")
        if labels_in_file and LIBXMP:
            self.labels = self._extract_labels(fn)
        else:
            self.labels = labels

        Track.__init__(self, fn, name)

    @property
    def analysis(self):
        """Get musical anaylsis of the song using the librosa library
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

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            self._labels = None
        else:
            self._labels = sorted(labels, key=lambda x: x.time)

    def label(self, t):
        """Get the label of the song at a given time in seconds
        """
        if self.labels is None:
            return None
        prev_label = None
        for l in self.labels:
            if l.time > t: break
            prev_label = l
        if prev_label is None: return None
        return prev_label.name

    def _extract_labels(self, filename):
        if not LIBXMP: return None

        xmp = libxmp.utils.file_to_dict(filename)
        meta = libxmp.XMPMeta()
        ns = libxmp.consts.XMP_NS_DM
        p = meta.get_prefix_for_namespace(ns)

        track_re = re.compile("^" + p + r"Tracks\[(\d+)\]$")
        n_tracks = 0
        cp_track = None
        new_xmp = {}
        for prop in xmp[ns]:
            new_xmp[prop[0]] = prop[1:]

        # find the cuepoint markers track
        name_re = re.compile("^" + p + r"Tracks\[(\d+)\]/" + p + "trackName$")
        for prop, val in new_xmp.iteritems():
            match = name_re.match(prop)
            if match:
                if val[0] == "CuePoint Markers":
                    cp_track = match.group(1)

        # get all the markers from it
        cp_path = re.compile(r"^%sTracks\[%s\]/%smarkers\[(\d+)\]$"  % (p, cp_track, p))
        markers = []
        sr = float(new_xmp["%sTracks[%s]/%sframeRate" % (p, cp_track, p)][0].replace('f', ''))

        for prop, val in new_xmp.iteritems():
            match = cp_path.match(prop)
            if match:
                markers.append(Label(
                    new_xmp[prop + '/' + p + 'name'][0],
                    float(new_xmp[prop + '/' + p + 'startTime'][0]) / sr))

        if len(markers) is 0:
            return None
        return markers


