radiotool - tools for constructing audio
========================================

Important note
--------------

Some parts of this codebase have had significant changes since the
last revision of the documentation- so the current documentation is
wrong in some places. I will try to update the documentation as soon
as I can, but for now, feel free to ask me questions if anything is
not clear.

Description
-----------

Radiotool is a python library that aims to make it easy to create
audio by piecing together bits of other audio files. This library was
originally written to enable my research in audio editing user
interfaces, but perhaps someone else might find it useful.

Read the full documentation_.

.. _documentation: http://ucbvislab.github.io/radiotool

To perform the actual audio rendering, radiotool relies on
scikits.audiolab_, a python wrapper for libsndfile_.

.. _scikits.audiolab: https://pypi.python.org/pypi/scikits.audiolab/
.. _libsndfile: http://www.mega-nerd.com/libsndfile/

Installation
------------

Either ``pip install radiotool`` or clone the repository and run
``python setup.py install``.

Requirements
------------

libsndfile_, numpy, scikits.audiolab_, and librosa_.

exempi_ for writing Audition/Premiere-readable labels to audio files.

.. _exempi: http://libopenraw.freedesktop.org/wiki/Exempi/
.. _librosa: https://github.com/bmcfee/librosa/

Composition
-----------
 
The heart of radiotool is the ``Composition``. A ``Composition`` is
built out of Segments, which represent segments of audio
Tracks (or raw PCM data, in the case of RawTracks). You can
also add Dynamics to adjust the volume of segments in certain
ways. 

Simple example
~~~~~~~~~~~~~~

.. code:: python

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

Retargeting
-----------

Music retargeting is the idea of taking a song and remixing it *from
its own existing beats/structure* to fit the music to certain
constraints.

See http://ucbvislab.github.io/radiotool/algorithms/retarget.html for
applications of music retargeting, and details about how to retarget
music using raditool.

See the documentation_ for more detail.