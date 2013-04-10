from track import Track

class Song(Track):
    """A :py:class:`radiotool.composer.Track`
    subclass that wraps a music .wav file.

    Nothing special here yet, but this will become more interesting in
    the future.
    """

    def __init__(self, fn, name="Song name"):
        Track.__init__(self, fn, name)

    