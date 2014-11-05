
from collections import namedtuple

import numpy as np


class VolumeBreakpoint(
        namedtuple('VolumeBreakpoint', ['time', 'volume', 'fade_type'])):
    def __new__(cls, time, volume, fade_type="exponential"):
        return super(VolumeBreakpoint, cls).__new__(
            cls, time, volume, fade_type)


class VolumeBreakpoints(object):
    def __init__(self, volume_breakpoints):
        self.breakpoints = volume_breakpoints

    def add_breakpoint(self, bp):
        self.breakpoints.append(bp)

    def add_breakpoints(self, bps):
        self.breakpoints.extend(bps)

    def to_array(self, samplerate):
        sorted_bps = sorted(self.breakpoints, key=lambda x: x.time)
        arr = np.ones(int(sorted_bps[-1][0] * samplerate))
        for i, bp in enumerate(sorted_bps[:-1]):
            t = int(bp.time * samplerate)
            v = bp.volume
            next_t = int(sorted_bps[i + 1].time * samplerate)
            next_v = sorted_bps[i + 1].volume

            if bp.fade_type == "exponential" and v != next_v:
                if v < next_v:
                    arr[t:next_t] = np.logspace(
                        8, 1, num=next_t - t, base=.5) *\
                        (next_v - v) / .5 + v
                else:
                    arr[t:next_t] = np.logspace(
                        1, 8, num=next_t - t, base=.5) *\
                        (v - next_v) / .5 + next_v
            else:
                arr[t:next_t] = np.linspace(v, next_v, num=next_t - t)

        return arr
