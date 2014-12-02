import os
import sys

from radiotool.composer import Song
from radiotool.algorithms import constraints as rt_constraints
from radiotool.algorithms import retarget as rt_retarget


def retarget_to_duration(song_filename, duration, start=True, end=True):
    song = Song(song_filename)

    duration = float(duration)

    constraints = [
        rt_constraints.TimbrePitchConstraint(
            context=0, timbre_weight=1.0, chroma_weight=1.0),
        rt_constraints.EnergyConstraint(penalty=.5),
        rt_constraints.MinimumLoopConstraint(8),
    ]

    extra = ''

    if start:
        constraints.append(
            rt_constraints.StartAtStartConstraint(padding=0))
        extra += 'Start'

    if end:
        constraints.append(
            rt_constraints.EndAtEndConstraint(padding=12))
        extra += 'End'

    comp, info = rt_retarget.retarget(
        [song], duration, constraints=[constraints],
        fade_in_len=None, fade_out_len=None)

    # force the new track to extend to the end of the song
    if end == "end":
        last_seg = sorted(
            comp.segments,
            key=lambda seg:
            seg.comp_location_in_seconds + seg.duration_in_seconds
        )[-1]
        last_seg.duration_in_seconds = (
            song.duration_in_seconds - last_seg.start_in_seconds)

    result_fn = "{}-{}-{}".format(
        os.path.splitext(song_filename)[0], str(duration), extra)

    # if end == "end":
    #     frames = comp.build(channels=song.channels)
    #     new_track = time_stretch(frames, song.samplerate, duration)
    #     comp = Composition(channels=song.channels)
    #     comp.add_segment(
    #         Segment(new_track, 0.0, 0.0, new_track.duration_in_seconds))

    comp.export(filename=result_fn,
                channels=song.channels,
                filetype='mp3')

    print info["transitions"]

    path_cost = info["path_cost"]
    total_nonzero_cost = []
    total_nonzero_points = []
    for node in path_cost:
        if float(node.name) > 0.0:
            total_nonzero_cost.append(float(node.name))
            total_nonzero_points.append(float(node.time))

    transitions = zip(total_nonzero_points, total_nonzero_cost)

    print "Retargeted track to {}.mp3".format(result_fn)

    print "Transitions:"
    for trans in transitions:
        print "Time {} Cost {}".format(round(trans[0], 1), round(trans[1], 2))

if __name__ == '__main__':
    song_fn = sys.argv[1]
    duration = float(sys.argv[2])

    retarget_to_duration(song_fn, duration)
