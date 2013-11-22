from unittest import TestCase
import unittest
import os

import numpy as N

from radiotool.composer.track import Track


class TestTrack(TestCase):

    def setUp(self):
        self.dirname = os.path.dirname(os.path.abspath(__file__))
        test_filename = os.path.join(self.dirname, "test.wav")
        self.track = Track(test_filename, "test_track")

    def tearDown(self):
        self.track = None

    def test_samplerate(self):
        assert self.track.samplerate == 44100

    def test_num_frames(self):
        assert self.track.duration == 88200

    def test_read_frames(self):
        self.track.current_frame = 1000
        frames_read = self.track.read_frames(2)
        frames_real = N.array([[-0.30187988, 0.24993896],
                               [-0.30731201, 0.26702881]])
        assert N.allclose(frames_read, frames_real)
        assert self.track.current_frame == 1002

    def test_read_mp3(self):
        mp3_filename = os.path.join(self.dirname, "test.mp3")
        with self.assertRaises(IOError) as ctx:
            Track(mp3_filename, "test mp3")
        # can use the ctx.exception.message later if desired

