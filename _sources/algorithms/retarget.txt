Music retargeting
=================

Music retargeting is the idea of taking a song and remixing it *from
its own existing beats/structure* to fit the music to certain
constraints.

Example applications:

* Use :func:`~radiotool.algorithms.retarget.retarget_to_length`
  to create a new shorter or longer version of a song that is a
  specific duration.
* Use
  :func:`~radiotool.algorithms.retarget.retarget_with_change_points` to
  reach music change points (see :ref:`novelty`) at specified points
  in the retargeted music
* The EchoNest's `Infinite Jukebox`_.

.. _Infinite Jukebox: http://labs.echonest.com/Uploader/index.html

.. autofunction:: radiotool.algorithms.retarget.retarget_to_length
.. autofunction:: radiotool.algorithms.retarget.retarget_with_change_points
.. autofunction:: radiotool.algorithms.retarget.retarget
