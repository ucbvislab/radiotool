.. radiotool documentation master file, created by
   sphinx-quickstart on Tue Apr  9 14:36:50 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

radiotool - tools for constructing and retargeting audio
========================================================

Radiotool is a python library that aims to make it easy to create
audio by piecing together bits of other audio files. This library was
originally written to enable my research in audio editing user
interfaces, but perhaps someone else might find it useful.


To perform the actual audio rendering, radiotool relies on
scikits.audiolab_, a python wrapper for libsndfile_.

.. _scikits.audiolab: https://pypi.python.org/pypi/scikits.audiolab/
.. _libsndfile: http://www.mega-nerd.com/libsndfile/

The library source is hosted on GitHub here_.

.. _here: https://github.com/ucbvislab/radiotool

Contents:

.. toctree::
   :maxdepth: 2

   composer/composition
   composer/tracks
   composer/segments
   composer/dynamics
   algorithms/retarget
   algorithms/novelty
   utils

.. currentmodule:: radiotool

Simple examples
---------------

Basic usage
~~~~~~~~~~~

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

Simple extension or shortening of music
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from radiotool.composer import Song
    from radiotool.algorithms.retarget import retarget_to_length

    song = Song("music_file.wav")

    # new song should be 1000 seconds long
    composition = retarget_to_length(song, 1000)

    # new song that's only 30 seconds long
    short_composition = retarget_to_length(song, 30)

    # export to audio file
    composition.export(filename="retarget_length_test")
    short_composition.export(filename="short_retarget_length_test")


See the documentation for more detailed examples (coming soon!).

Composition
-----------

.. autosummary:: 
    radiotool.composer.Composition

Tracks
------

.. autosummary::
    radiotool.composer.Track
    radiotool.composer.Speech
    radiotool.composer.Song
    radiotool.composer.RawTrack

Segments
--------

.. autosummary::
    radiotool.composer.Segment
    radiotool.composer.TimeStretchSegment

Dynamics
--------

.. autosummary::
  radiotool.composer.Dynamic
  radiotool.composer.Volume
  radiotool.composer.Fade
  radiotool.composer.RawVolume
  radiotool.composer.VolumeBreakpoint

Algorithms
----------

.. autosummary::
  radiotool.algorithms.retarget.retarget_to_length
  radiotool.algorithms.retarget.retarget_with_change_points
  radiotool.algorithms.retarget.retarget
  radiotool.algorithms.novelty

Utilities
---------

.. autosummary::
    radiotool.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

