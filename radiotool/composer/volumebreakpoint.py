
from collections import namedtuple

import numpy as N

VolumeBreakpoint = namedtuple('VolumeBreakpoint', ['time', 'volume'])

class VolumeBreakpoints(object):
    def __init__(self, volume_breakpoints):
        self.breakpoints = volume_breakpoints

    def add_breakpoint(self, bp):
        self.breakpoints.append(bp)

    def add_breakpoints(self, bps):
        self.breakpoints.extend(bps)

    def to_array(self, samplerate):
        sorted_bps = sorted(self.breakpoints, key=lambda x: x.time)
        arr = N.ones(int(sorted_bps[-1][0] * samplerate))
        for i, bp in enumerate(sorted_bps[:-1]):
            t = int(bp.time * samplerate)
            v = bp.volume
            next_t = int(sorted_bps[i + 1].time * samplerate)
            next_v = sorted_bps[i + 1].volume
            arr[t:next_t] = N.linspace(v, next_v, num=next_t - t)
        return arr
