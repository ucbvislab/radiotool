radiotool - tools for constructing audio
========================================

Radiotool is a python library that aims to make it easy to create
audio by piecing together bits of other audio files. This library was
originally written to enable my research in audio editing user
interfaces, but perhaps someone else might find it useful.

To perform the actual audio rendering, radiotool relies on
scikits.audiolab_, a python wrapper for libsndfile_.

.. _scikits.audiolab: https://pypi.python.org/pypi/scikits.audiolab/
.. _libsndfile: http://www.mega-nerd.com/libsndfile/

Composition
-----------
 
The heart of radiotool is the ``Composition``. ``Composition``s are
built out of ``Segment``s, which represent segments of audio
``Track``s (or raw PCM data, in the case of ``RawTrack``s). You can
also add ``Dynamic``s to adjust the volume of segments in certain
ways.

Simple example
~~~~~~~~~~~~~~

::

    from radiotool.composer import *
    comp = Composition()
    
    # create a track with a pre-existing wav file
    track = Track("test.wav")

    # create a segment of a track that:
    # 1. starts at the 0.0 mark of the composition
    # 2. begins playing at the 0.5 second mark of the track
    # 3. plays for 1.0 seconds
    segment = Segment(track, 0.0, 0.5, 1.0)

    # add segment to the composition
    comp.add_segment(segment)

    # output your composition as a numpy array
    arr_out = comp.build()

    # or export your composition as an audio file, composition.wav
    comp.export(filename="composition")

See the documentation (coming soon!) for more detailed examples.